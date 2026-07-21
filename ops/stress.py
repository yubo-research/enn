#!/usr/bin/env python

from __future__ import annotations

import json
import resource
import struct
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.enn_params import ENNParams, PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver

INDEX_TYPE_CHOICES: tuple[str, ...] = ("flat", "bpann_disk")
DISK_INDEX_TYPE_CHOICES: frozenset[str] = frozenset({"bpann_disk"})
DISK_DEFER_SYNC_DRIVERS: frozenset[ENNIndexDriver] = frozenset(
    {ENNIndexDriver.BPANN_DISK}
)
DEFAULT_NUM_DIM = 10
STRESS_OBS_BATCH_SIZE = 100
ENN_ADD_STRESS_BATCH_SIZE = 1000
DEFAULT_HEARTBEAT_SECONDS = 10.0
STRESS_QUERY_N = 1000
STRESS_QUERY_SEED = 1
STRESS_QUERY_K = 10
NUM_SAMPLE = STRESS_QUERY_N
DEFAULT_SAMPLE_WORK_DIR = "_enn"
DEFAULT_SAMPLE_X_LOW = -1.0
DEFAULT_SAMPLE_X_HIGH = 1.0
DEFAULT_DRAW_SEED = 0
DEFAULT_DRAW_K = 10
DEFAULT_DRAW_NUM_FIT_CANDIDATES = 30
DEFAULT_DRAW_NUM_FIT_SAMPLES = 10
DEFAULT_DRAW_NUM_DRAWS = 100
DEFAULT_DRAW_NUM_SEEDS = 1
DRAW_OBS_NOISE_STD = 0.1
DRAW_F_CENTER = 0.3
DRAW_FLAGS = PosteriorFlags(observation_noise=True)
DRAW_FLAGS_NO_OBS = PosteriorFlags(observation_noise=False)
STRESS_PARAMS = ENNParams(
    k_num_neighbors=STRESS_QUERY_K,
    epistemic_variance_scale=1.0,
    aleatoric_variance_scale=0.1,
)
DISK_STRESS_RSS_BASELINE_MIB = 512
DISK_STRESS_RSS_PER_SHARD_ROW_BYTES = 64
DEFAULT_SHARD_MAX_ROWS = 500_000
TURBO_ENN_NUM_INIT = 10
TURBO_ENN_ACKLEY_NOISE = 0.1
TURBO_ENN_SEED = 0
TURBO_ENN_K = 10
TURBO_ENN_NUM_FIT_SAMPLES = 100
# Larger than proposal-scale chunk: turbo-enn large gaps pay per-tell fit/TR cost.
# Fewer, larger tells cut soft-sync overhead on bpann_disk empty-leaf drains.
TURBO_ENN_SEED_CHUNK = 100_000
TURBO_ENN_SEED_CHUNK_FLAT = 200_000
PROPOSAL_SCALE_NS: tuple[int, ...] = (
    10,
    30,
    100,
    300,
    1000,
    3000,
    10000,
    30000,
    100000,
)
PROPOSAL_SCALE_NUM_PROBES = 30
PROPOSAL_SCALE_WARMUP = 2
PROPOSAL_SCALE_SEED_CHUNK = 1000


def max_rss_bytes() -> int:
    """Peak resident set size in bytes (platform-normalized)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def disk_stress_rss_ceiling_bytes(
    *, num_dim: int, shard_max_rows: int = DEFAULT_SHARD_MAX_ROWS
) -> int:
    """Acceptance #3 RSS delta ceiling from plan."""
    return (
        DISK_STRESS_RSS_BASELINE_MIB * 1024 * 1024
        + DISK_STRESS_RSS_PER_SHARD_ROW_BYTES * num_dim * shard_max_rows
    )


@dataclass(frozen=True)
class DiskRssStressConfig:
    num_dim: int = DEFAULT_NUM_DIM
    seed: int = 0
    batch_size: int = STRESS_OBS_BATCH_SIZE
    query_n: int = STRESS_QUERY_N


@dataclass(frozen=True)
class DiskRssStressResult:
    num_obs: int
    baseline_rss_bytes: int
    final_rss_bytes: int
    rss_delta_bytes: int
    train_x_bytes: int
    index_memory_bytes: int


def run_disk_rss_stress(
    *,
    num_obs: int,
    work_dir: str,
    index_driver: ENNIndexDriver = ENNIndexDriver.BPANN_DISK,
    config: DiskRssStressConfig | None = None,
) -> DiskRssStressResult:
    """Stream batched adds to disk ENN; return RSS and on-disk metrics."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    cfg = config if config is not None else DiskRssStressConfig()
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if cfg.query_n < 1:
        raise ValueError("query_n must be >= 1")
    baseline_rss = max_rss_bytes()
    x_query = make_query_points(
        cfg.query_n, num_dim=cfg.num_dim, seed=STRESS_QUERY_SEED
    )

    empty_x = np.empty((0, cfg.num_dim), dtype=float)
    empty_y = np.empty((0, 1), dtype=float)
    model = EpistemicNearestNeighbors(
        empty_x,
        empty_y,
        scale_x=False,
        index_driver=index_driver,
        work_dir=work_dir,
        enn_storage="disk",
    )
    baseline_after_init = max_rss_bytes()

    rows_added = 0
    for x_batch, y_batch in iter_synthetic_observation_batches(
        num_obs,
        num_dim=cfg.num_dim,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
    ):
        model.add(x_batch, y_batch)
        rows_added += x_batch.shape[0]
        model.schedule_background_flush()
    assert rows_added == num_obs

    model.ensure_index_sync()
    posterior = model.posterior(x_query, params=STRESS_PARAMS)
    assert np.all(np.isfinite(posterior.mu))
    assert np.all(np.isfinite(posterior.se))

    final_rss = max_rss_bytes()
    train_x_path = Path(work_dir) / "train_x.bin"
    train_x_bytes = train_x_path.stat().st_size if train_x_path.exists() else 0
    expected_train_x = num_obs * cfg.num_dim * 8
    assert abs(train_x_bytes - expected_train_x) <= cfg.num_dim * 8
    assert len(model) == num_obs

    return DiskRssStressResult(
        num_obs=num_obs,
        baseline_rss_bytes=min(baseline_rss, baseline_after_init),
        final_rss_bytes=final_rss,
        rss_delta_bytes=final_rss - min(baseline_rss, baseline_after_init),
        train_x_bytes=train_x_bytes,
        index_memory_bytes=model.index_memory_bytes(),
    )


@dataclass(frozen=True)
class EnnAddStressConfig:
    num_dim: int = DEFAULT_NUM_DIM
    seed: int = 0
    batch_size: int = 1
    progress_every: int = 0
    heartbeat_seconds: float = 0.0
    query_n: int = STRESS_QUERY_N
    query_seed: int = STRESS_QUERY_SEED
    work_dir: str | None = None


def parse_index_driver(name: str) -> ENNIndexDriver:
    mapping = {
        "flat": ENNIndexDriver.FLAT,
        "bpann_disk": ENNIndexDriver.BPANN_DISK,
    }
    if name not in mapping:
        raise ValueError(f"Unknown index type: {name}")
    return mapping[name]


def turbo_enn_default_seed_chunk(index_driver: ENNIndexDriver) -> int:
    """Return turbo-enn bulk-seed chunk sized for the index backend."""
    if index_driver == ENNIndexDriver.FLAT:
        return TURBO_ENN_SEED_CHUNK_FLAT
    return TURBO_ENN_SEED_CHUNK


def _next_checkpoint(n: int) -> int:
    if n < 3:
        return 3
    if n < 10:
        return 10
    if n % 30 == 10:
        return n * 3
    return n * 10 // 3


def checkpoint_ns(max_n: int) -> tuple[int, ...]:
    """Return checkpoint sizes N=1, 3, 10, 30, 100, ... up to max_n."""
    if max_n < 1:
        raise ValueError("max_n must be >= 1")
    out: list[int] = []
    n = 1
    while n <= max_n:
        out.append(n)
        n = _next_checkpoint(n)
    return tuple(out)


def make_synthetic_observations(
    num_obs: int, *, num_dim: int = 10, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_obs, num_dim))
    y = rng.standard_normal((num_obs, 1))
    return x, y


def make_query_points(query_n: int, *, num_dim: int, seed: int) -> np.ndarray:
    """Return (query_n, num_dim) query batch held constant across checkpoints."""
    if query_n < 1:
        raise ValueError("query_n must be >= 1")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((query_n, num_dim))


def make_uniform_query_points(
    query_n: int,
    *,
    num_dim: int,
    seed: int,
    low: float = DEFAULT_SAMPLE_X_LOW,
    high: float = DEFAULT_SAMPLE_X_HIGH,
) -> np.ndarray:
    """Return (query_n, num_dim) uniformly random query batch in [low, high]."""
    if query_n < 1:
        raise ValueError("query_n must be >= 1")
    if num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    if low >= high:
        raise ValueError("low must be < high")
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(query_n, num_dim))


def iter_synthetic_observations(
    num_obs: int,
    *,
    num_dim: int = DEFAULT_NUM_DIM,
    seed: int = 0,
    batch_size: int = STRESS_OBS_BATCH_SIZE,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (1, num_dim) x and (1, 1) y rows without holding all num_obs rows."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    rng = np.random.default_rng(seed)
    emitted = 0
    while emitted < num_obs:
        n = min(batch_size, num_obs - emitted)
        for _ in range(n):
            x_row = rng.standard_normal((1, num_dim))
            y_row = rng.standard_normal((1, 1))
            yield x_row, y_row
        emitted += n


def iter_synthetic_observation_batches(
    num_obs: int,
    *,
    num_dim: int = DEFAULT_NUM_DIM,
    seed: int = 0,
    batch_size: int = STRESS_OBS_BATCH_SIZE,
    stop_points: Sequence[int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (n, num_dim) x and (n, 1) y batches without holding all num_obs rows.

    When ``stop_points`` is given, no batch crosses a stop point, so cumulative
    row counts land exactly on each stop point (checkpoint alignment).
    """
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    stops = sorted(s for s in (stop_points or ()) if 1 <= s <= num_obs)
    stop_i = 0
    rng = np.random.default_rng(seed)
    emitted = 0
    while emitted < num_obs:
        n = min(batch_size, num_obs - emitted)
        while stop_i < len(stops) and stops[stop_i] <= emitted:
            stop_i += 1
        if stop_i < len(stops):
            n = min(n, stops[stop_i] - emitted)
        x_batch = rng.standard_normal((n, num_dim))
        y_batch = rng.standard_normal((n, 1))
        yield x_batch, y_batch
        emitted += n


def load_disk_metadata(work_dir: str) -> dict:
    """Return metadata.json for a disk-backed bpann store."""
    meta_path = Path(work_dir) / "metadata.json"
    if not meta_path.is_file():
        raise ValueError(f"metadata.json not found in {work_dir}")
    meta = json.loads(meta_path.read_text())
    if meta.get("index_backend") != "bpann_disk":
        raise ValueError(
            f"work_dir index_backend is not bpann_disk: {meta.get('index_backend')!r}"
        )
    for key in ("num_dim", "num_metrics", "num_obs"):
        if key not in meta:
            raise ValueError(f"metadata.json missing {key!r}")
    return meta


def function_seeds_for_sample(seed: int, num_seeds: int = 1) -> list[int]:
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    return list(range(seed, seed + num_seeds))


def reopen_disk_bpann_enn(work_dir: str) -> tuple[EpistemicNearestNeighbors, dict]:
    """Reopen a persisted bpann_disk store from work_dir."""
    meta = load_disk_metadata(work_dir)
    num_dim = int(meta["num_dim"])
    num_metrics = int(meta["num_metrics"])
    scale_x = bool(meta.get("scale_x", False))
    model = EpistemicNearestNeighbors(
        np.empty((0, num_dim), dtype=float),
        np.empty((0, num_metrics), dtype=float),
        scale_x=scale_x,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=work_dir,
        enn_storage="disk",
    )
    model.ensure_index_sync()
    return model, meta


def load_num_obs_existing(work_dir: str) -> int:
    """Return persisted observation count in work_dir, or 0 if none."""
    root = Path(work_dir)
    sidecar = root / "num_obs.bin"
    if sidecar.is_file() and sidecar.stat().st_size == 8:
        return int(struct.unpack("<Q", sidecar.read_bytes())[0])
    meta_path = root / "metadata.json"
    if meta_path.is_file():
        try:
            num_obs = json.loads(meta_path.read_text())["num_obs"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return 0
        if isinstance(num_obs, int) and num_obs >= 0:
            return num_obs
    return 0


def format_config_header(
    *,
    num_dim: int,
    num_obs: int,
    work_dir: str | None = None,
    num_obs_existing: int | None = None,
) -> str:
    prefix = "restarting " if num_obs_existing else ""
    header = f"{prefix}num_dim={num_dim} num_obs={num_obs}"
    if num_obs_existing:
        header = f"{header} num_obs_existing={num_obs_existing}"
    if work_dir is not None:
        header = f"{header} work_dir={work_dir}"
    return header


def _time_query_s(model: EpistemicNearestNeighbors, x_query: np.ndarray) -> float:
    t0 = time.perf_counter()
    model.posterior(x_query, params=STRESS_PARAMS)
    return time.perf_counter() - t0


def run_enn_add_stress(
    *,
    index_driver: ENNIndexDriver,
    num_obs: int,
    config: EnnAddStressConfig | None = None,
) -> Iterator[tuple[int, float, float]]:
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    cfg = config if config is not None else EnnAddStressConfig()
    checkpoints = set(checkpoint_ns(num_obs))
    x_query = make_query_points(cfg.query_n, num_dim=cfg.num_dim, seed=cfg.query_seed)

    empty_x = np.empty((0, cfg.num_dim), dtype=float)
    empty_y = np.empty((0, 1), dtype=float)
    model_kwargs: dict[str, object] = {
        "train_x": empty_x,
        "train_y": empty_y,
        "scale_x": False,
        "index_driver": index_driver,
    }
    if cfg.work_dir is not None:
        model_kwargs["work_dir"] = cfg.work_dir
        model_kwargs["enn_storage"] = "disk"
    model = EpistemicNearestNeighbors(**model_kwargs)
    # Zero-row warmup: warms the Python validation/binding path only (the
    # library short-circuits on zero rows), so checkpoint 1 excludes
    # interpreter warmup but still pays first-touch library costs.
    model.add(empty_x, empty_y)

    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    checkpoint_list = sorted(checkpoints)
    next_checkpoint_i = 0
    last_heartbeat_t = time.perf_counter()
    # Library-only time (model.add + flush scheduling) accumulated since the
    # previous checkpoint; excludes synthetic-data generation and loop overhead.
    segment_lib_s = 0.0

    def emit_checkpoints(current_n: int) -> Iterator[tuple[int, float, float]]:
        nonlocal next_checkpoint_i, segment_lib_s
        while (
            next_checkpoint_i < len(checkpoint_list)
            and checkpoint_list[next_checkpoint_i] <= current_n
        ):
            cp = checkpoint_list[next_checkpoint_i]
            # Index sync is library work, so it counts toward the segment.
            # Disk mode (batch or single-row) defers indexing to
            # schedule_background_flush inside the add loop, matching the
            # baseline; forcing a sync here would build tiny index fragments
            # the baseline never built.
            if index_driver not in DISK_DEFER_SYNC_DRIVERS:
                t_sync = time.perf_counter()
                model.ensure_index_sync()
                segment_lib_s += time.perf_counter() - t_sync
            segment_s = segment_lib_s
            segment_lib_s = 0.0
            query_s = _time_query_s(model, x_query)
            yield (cp, query_s, segment_s)
            next_checkpoint_i += 1

    try:
        n = 0
        if cfg.batch_size <= 1:
            row_iter = iter_synthetic_observations(
                num_obs, num_dim=cfg.num_dim, seed=cfg.seed
            )
            while True:
                try:
                    x_row, y_row = next(row_iter)
                except StopIteration:
                    break
                n += 1
                t_add = time.perf_counter()
                model.add(x_row, y_row)
                if index_driver in DISK_DEFER_SYNC_DRIVERS:
                    model.schedule_background_flush()
                segment_lib_s += time.perf_counter() - t_add
                if cfg.progress_every and (n % cfg.progress_every == 0):
                    click.echo(f"progress n={n}", err=True)
                if cfg.heartbeat_seconds and (
                    time.perf_counter() - last_heartbeat_t >= cfg.heartbeat_seconds
                ):
                    click.echo(f"heartbeat n={n}", err=True)
                    last_heartbeat_t = time.perf_counter()
                yield from emit_checkpoints(n)
        else:
            for x_batch, y_batch in iter_synthetic_observation_batches(
                num_obs,
                num_dim=cfg.num_dim,
                seed=cfg.seed,
                batch_size=cfg.batch_size,
                stop_points=checkpoint_list,
            ):
                t_add = time.perf_counter()
                model.add(x_batch, y_batch)
                if index_driver in DISK_DEFER_SYNC_DRIVERS:
                    model.schedule_background_flush()
                segment_lib_s += time.perf_counter() - t_add
                n += x_batch.shape[0]
                if cfg.progress_every and (n % cfg.progress_every == 0):
                    click.echo(f"progress n={n}", err=True)
                if cfg.heartbeat_seconds and (
                    time.perf_counter() - last_heartbeat_t >= cfg.heartbeat_seconds
                ):
                    click.echo(f"heartbeat n={n}", err=True)
                    last_heartbeat_t = time.perf_counter()
                yield from emit_checkpoints(n)
    finally:
        if index_driver in DISK_DEFER_SYNC_DRIVERS:
            model.persist_index_to_disk()


def stress_row_n_width(num_obs: int) -> int:
    """Character width for the N column; sized for the largest checkpoint (num_obs)."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    return len(str(num_obs))


def format_stress_row(n: int, query_s: float, segment_s: float, *, n_width: int) -> str:
    return f"{n:>{n_width}} {query_s:.3f} {segment_s:.3g}"


@dataclass(frozen=True)
class SampleStressConfig:
    num_samples: int = NUM_SAMPLE
    seed: int = STRESS_QUERY_SEED
    x_low: float = DEFAULT_SAMPLE_X_LOW
    x_high: float = DEFAULT_SAMPLE_X_HIGH


@dataclass(frozen=True)
class SampleStressResult:
    num_dim: int
    num_obs: int
    num_samples: int
    seed: int
    num_function_seeds: int
    draws_shape: tuple[int, ...]
    all_finite: bool
    init_s: float
    sample_s: float


def run_sample_stress(
    *,
    work_dir: str,
    config: SampleStressConfig | None = None,
) -> SampleStressResult:
    """Reopen disk ENN and draw posterior function samples at uniform random x."""
    cfg = config if config is not None else SampleStressConfig()
    if cfg.num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    t0 = time.perf_counter()
    model, meta = reopen_disk_bpann_enn(work_dir)
    num_dim = int(meta["num_dim"])
    x_query = make_uniform_query_points(
        cfg.num_samples,
        num_dim=num_dim,
        seed=cfg.seed,
        low=cfg.x_low,
        high=cfg.x_high,
    )
    init_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    function_seeds = function_seeds_for_sample(cfg.seed)
    draws, _idx = model.posterior_function_draw(
        x_query,
        STRESS_PARAMS,
        function_seeds=function_seeds,
    )
    sample_s = time.perf_counter() - t1
    return SampleStressResult(
        num_dim=num_dim,
        num_obs=len(model),
        num_samples=cfg.num_samples,
        seed=cfg.seed,
        num_function_seeds=len(function_seeds),
        draws_shape=tuple(int(s) for s in draws.shape),
        all_finite=bool(np.all(np.isfinite(draws))),
        init_s=init_s,
        sample_s=sample_s,
    )


def format_sample_config_header(*, result: SampleStressResult, work_dir: str) -> str:
    return (
        f"num_dim={result.num_dim} num_obs={result.num_obs} "
        f"work_dir={work_dir} num_samples={result.num_samples} seed={result.seed}"
    )


def format_sample_summary(result: SampleStressResult) -> str:
    return (
        f"draws_shape={result.draws_shape} function_seeds={result.num_function_seeds} "
        f"all_finite={str(result.all_finite).lower()} init_s={result.init_s:.3f} "
        f"sample_s={result.sample_s:.3f}"
    )


def draw_f(x: np.ndarray) -> np.ndarray:
    """Return sum_j (x_ij - DRAW_F_CENTER)^2 with shape (n, 1)."""
    x_arr = np.asarray(x, dtype=float)
    return np.sum((x_arr - DRAW_F_CENTER) ** 2, axis=1, keepdims=True)


def _draw_argmin_indices(x_test: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Return per-sample argmin indices over metric 0; ``draws`` is ``(B, M, S)``."""
    x_arr = np.asarray(x_test, dtype=float)
    draws_arr = np.asarray(draws, dtype=float)
    if draws_arr.ndim != 3:
        raise ValueError(f"draws must be 3D (B, M, S), got shape {draws_arr.shape}")
    if draws_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(
            f"x_test rows ({x_arr.shape[0]}) must match draws batch ({draws_arr.shape[0]})"
        )
    if draws_arr.shape[1] < 1:
        raise ValueError("draws must have at least one metric")
    if draws_arr.shape[2] < 1:
        raise ValueError("draws must have at least one sample")
    return np.argmin(draws_arr[:, 0, :], axis=0)


def argmin_rms(x_test: np.ndarray, draws: np.ndarray) -> float:
    """RMS of ||x_hat - DRAW_F_CENTER||_2 over draws; x_hat = argmin of metric 0.

    ``draws`` has shape ``(batch, metrics, num_samples)``. For each sample ``s``,
    ``i* = argmin_i draws[i, 0, s]`` and ``x_hat = x_test[i*]``.
    """
    x_arr = np.asarray(x_test, dtype=float)
    i_star = _draw_argmin_indices(x_test, draws)
    eps = x_arr[i_star] - DRAW_F_CENTER
    return float(np.sqrt(np.mean(np.sum(eps * eps, axis=-1))))


def argmin_hit_rate(x_test: np.ndarray, draws: np.ndarray) -> float:
    """Fraction of draws whose metric-0 argmin matches ``argmin_i draw_f(x_test)_i``.

    ``draws`` has shape ``(batch, metrics, num_samples)``.
    """
    i_true = int(np.argmin(draw_f(x_test)[:, 0]))
    i_star = _draw_argmin_indices(x_test, draws)
    return float(np.mean(i_star == i_true))


def make_draw_observations(
    num_obs: int,
    *,
    num_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw x ~ U[0,1]^num_dim and y = draw_f(x) + 0.1 N(0,1)."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    x = rng.uniform(0.0, 1.0, size=(num_obs, num_dim))
    noise = rng.standard_normal((num_obs, 1))
    y = draw_f(x) + DRAW_OBS_NOISE_STD * noise
    return x, y


def gaussian_likelihood(y: np.ndarray, mu: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Per-point N(y; mu, se^2) densities."""
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    se_arr = np.asarray(se, dtype=float)
    z = (y_arr - mu_arr) / se_arr
    return (1.0 / (se_arr * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def average_likelihood(y: np.ndarray, mu: np.ndarray, se: np.ndarray) -> float:
    """Mean of per-point Gaussian predictive densities."""
    return float(np.mean(gaussian_likelihood(y, mu, se)))


def average_likelihood_from_draws(y: np.ndarray, draws: np.ndarray) -> float:
    """Mean Gaussian density using empirical mean/std over sample axis -1.

    ``draws`` has shape ``(batch, metrics, num_samples)``.
    """
    draws_arr = np.asarray(draws, dtype=float)
    if draws_arr.ndim != 3:
        raise ValueError(f"draws must be 3D (B, M, S), got shape {draws_arr.shape}")
    if draws_arr.shape[-1] < 2:
        raise ValueError("draws must have at least 2 samples along axis -1")
    mu = draws_arr.mean(axis=-1)
    se = draws_arr.std(axis=-1, ddof=1)
    se = np.maximum(se, 1e-12)
    return average_likelihood(y, mu, se)


@dataclass(frozen=True)
class DrawStressConfig:
    num_obs: int
    num_test: int
    num_dim: int = DEFAULT_NUM_DIM
    seed: int = DEFAULT_DRAW_SEED
    k: int = DEFAULT_DRAW_K
    num_fit_candidates: int = DEFAULT_DRAW_NUM_FIT_CANDIDATES
    num_fit_samples: int = DEFAULT_DRAW_NUM_FIT_SAMPLES
    num_draws: int = DEFAULT_DRAW_NUM_DRAWS


@dataclass(frozen=True)
class DrawMethodResult:
    method: str
    avg_likelihood: float
    argmin_rms: float
    argmin_hit_rate: float
    draws_shape: tuple[int, ...]
    all_finite: bool
    eval_s: float


@dataclass(frozen=True)
class DrawStressResult:
    num_obs: int
    num_test: int
    num_dim: int
    seed: int
    k: int
    num_fit_candidates: int
    num_fit_samples: int
    num_draws: int
    epistemic_variance_scale: float
    aleatoric_variance_scale: float
    fit_s: float
    posterior: DrawMethodResult
    posterior_function_draw: DrawMethodResult


def run_draw_stress(config: DrawStressConfig) -> DrawStressResult:
    """Fit ENN on synthetic draw data; score avg likelihood and argmin-RMS."""
    if config.num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if config.num_test < 1:
        raise ValueError("num_test must be >= 1")
    if config.num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    if config.k < 1:
        raise ValueError("k must be >= 1")
    if config.num_fit_candidates < 1:
        raise ValueError("num_fit_candidates must be >= 1")
    if config.num_fit_samples < 1:
        raise ValueError("num_fit_samples must be >= 1")
    if config.num_draws < 2:
        raise ValueError("num_draws must be >= 2")

    data_rng = np.random.default_rng(config.seed)
    fit_rng = np.random.default_rng(config.seed + 1)
    sample_rng = np.random.default_rng(config.seed + 2)
    x, y = make_draw_observations(config.num_obs, num_dim=config.num_dim, rng=data_rng)
    x_test, y_test = make_draw_observations(
        config.num_test, num_dim=config.num_dim, rng=data_rng
    )
    model = EpistemicNearestNeighbors(x, y, scale_x=False)

    t0 = time.perf_counter()
    fitted = enn_fit(
        model,
        k=config.k,
        num_fit_candidates=config.num_fit_candidates,
        num_fit_samples=config.num_fit_samples,
        rng=fit_rng,
    )
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    # Likelihood path: observation_noise=True (posterior lik is analytic).
    post_lik = model.posterior(x_test, params=fitted, flags=DRAW_FLAGS)
    avg_lik_post = average_likelihood(y_test, post_lik.mu, post_lik.se)
    # Argmin-RMS path: observation_noise=False joint draws.
    post_rms = model.posterior(x_test, params=fitted, flags=DRAW_FLAGS_NO_OBS)
    post_rms_draws = post_rms.sample(config.num_draws, sample_rng)
    post_argmin_rms = argmin_rms(x_test, post_rms_draws)
    post_argmin_hit_rate = argmin_hit_rate(x_test, post_rms_draws)
    eval_post_s = time.perf_counter() - t1
    posterior_result = DrawMethodResult(
        method="posterior",
        avg_likelihood=avg_lik_post,
        argmin_rms=post_argmin_rms,
        argmin_hit_rate=post_argmin_hit_rate,
        draws_shape=tuple(int(s) for s in post_rms_draws.shape),
        all_finite=bool(np.all(np.isfinite(post_rms_draws))),
        eval_s=eval_post_s,
    )

    t2 = time.perf_counter()
    # One ON=False joint draw serves both empirical lik and argmin metrics.
    # (A second ON=True draw nearly doubled eval_s without helping argmin quality.)
    function_seeds = function_seeds_for_sample(config.seed + 4, config.num_draws)
    fn_draws, _idx = model.posterior_function_draw(
        x_test,
        fitted,
        function_seeds=function_seeds,
        flags=DRAW_FLAGS_NO_OBS,
    )
    avg_lik_fn = average_likelihood_from_draws(y_test, fn_draws)
    fn_argmin_rms = argmin_rms(x_test, fn_draws)
    fn_argmin_hit_rate = argmin_hit_rate(x_test, fn_draws)
    eval_fn_s = time.perf_counter() - t2
    function_result = DrawMethodResult(
        method="posterior_function_draw",
        avg_likelihood=avg_lik_fn,
        argmin_rms=fn_argmin_rms,
        argmin_hit_rate=fn_argmin_hit_rate,
        draws_shape=tuple(int(s) for s in fn_draws.shape),
        all_finite=bool(np.all(np.isfinite(fn_draws))),
        eval_s=eval_fn_s,
    )

    return DrawStressResult(
        num_obs=config.num_obs,
        num_test=config.num_test,
        num_dim=config.num_dim,
        seed=config.seed,
        k=config.k,
        num_fit_candidates=config.num_fit_candidates,
        num_fit_samples=config.num_fit_samples,
        num_draws=config.num_draws,
        epistemic_variance_scale=float(fitted.epistemic_variance_scale),
        aleatoric_variance_scale=float(fitted.aleatoric_variance_scale),
        fit_s=fit_s,
        posterior=posterior_result,
        posterior_function_draw=function_result,
    )


@dataclass(frozen=True)
class MeanSE:
    mean: float
    se: float


@dataclass(frozen=True)
class DrawMethodAggregate:
    method: str
    avg_likelihood: MeanSE
    argmin_rms: MeanSE
    argmin_hit_rate: MeanSE
    eval_s: MeanSE


@dataclass(frozen=True)
class DrawStressAggregate:
    num_obs: int
    num_test: int
    num_dim: int
    seed: int
    num_seeds: int
    k: int
    num_fit_candidates: int
    num_fit_samples: int
    num_draws: int
    epistemic_variance_scale: MeanSE
    aleatoric_variance_scale: MeanSE
    fit_s: MeanSE
    posterior: DrawMethodAggregate
    posterior_function_draw: DrawMethodAggregate


def mean_se(values: np.ndarray | list[float]) -> MeanSE:
    """Sample mean and standard error of the mean (nan SE if fewer than 2 values)."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 1:
        raise ValueError("values must be non-empty")
    mean = float(np.mean(arr))
    if arr.size < 2:
        return MeanSE(mean=mean, se=float("nan"))
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return MeanSE(mean=mean, se=se)


def format_mean_se(stat: MeanSE, *, fmt: str = ".6g") -> str:
    if not np.isfinite(stat.se):
        return format(stat.mean, fmt)
    return f"{stat.mean:{fmt}} ± {stat.se:{fmt}}"


def _aggregate_draw_method(
    method: str, results: list[DrawMethodResult]
) -> DrawMethodAggregate:
    return DrawMethodAggregate(
        method=method,
        avg_likelihood=mean_se([r.avg_likelihood for r in results]),
        argmin_rms=mean_se([r.argmin_rms for r in results]),
        argmin_hit_rate=mean_se([r.argmin_hit_rate for r in results]),
        eval_s=mean_se([r.eval_s for r in results]),
    )


def run_draw_stress_over_seeds(
    config: DrawStressConfig, *, num_seeds: int
) -> DrawStressAggregate:
    """Run ``run_draw_stress`` for ``seed .. seed+num_seeds-1`` and aggregate metrics."""
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    results = [
        run_draw_stress(
            DrawStressConfig(
                num_obs=config.num_obs,
                num_test=config.num_test,
                num_dim=config.num_dim,
                seed=config.seed + i,
                k=config.k,
                num_fit_candidates=config.num_fit_candidates,
                num_fit_samples=config.num_fit_samples,
                num_draws=config.num_draws,
            )
        )
        for i in range(num_seeds)
    ]
    return DrawStressAggregate(
        num_obs=config.num_obs,
        num_test=config.num_test,
        num_dim=config.num_dim,
        seed=config.seed,
        num_seeds=num_seeds,
        k=config.k,
        num_fit_candidates=config.num_fit_candidates,
        num_fit_samples=config.num_fit_samples,
        num_draws=config.num_draws,
        epistemic_variance_scale=mean_se([r.epistemic_variance_scale for r in results]),
        aleatoric_variance_scale=mean_se([r.aleatoric_variance_scale for r in results]),
        fit_s=mean_se([r.fit_s for r in results]),
        posterior=_aggregate_draw_method("posterior", [r.posterior for r in results]),
        posterior_function_draw=_aggregate_draw_method(
            "posterior_function_draw", [r.posterior_function_draw for r in results]
        ),
    )


def format_draw_config_header(result: DrawStressResult) -> str:
    return (
        f"num_dim={result.num_dim} num_obs={result.num_obs} "
        f"num_test={result.num_test} seed={result.seed} k={result.k} "
        f"num_draws={result.num_draws} "
        f"epistemic_variance_scale={result.epistemic_variance_scale:.6g} "
        f"aleatoric_variance_scale={result.aleatoric_variance_scale:.6g} "
        f"fit_s={result.fit_s:.3f}"
    )


def format_draw_method_summary(method: DrawMethodResult) -> str:
    return (
        f"{method.method} avg_likelihood={method.avg_likelihood:.6g} "
        f"argmin_rms={method.argmin_rms:.6g} "
        f"argmin_hit_rate={method.argmin_hit_rate:0.4f} "
        f"eval_s={method.eval_s:.3f}"
    )


def format_draw_config_header_aggregate(result: DrawStressAggregate) -> str:
    return (
        f"num_dim={result.num_dim} num_obs={result.num_obs} "
        f"num_test={result.num_test} seed={result.seed} "
        f"num_seeds={result.num_seeds} k={result.k} "
        f"num_draws={result.num_draws} "
        f"epistemic_variance_scale={format_mean_se(result.epistemic_variance_scale)} "
        f"aleatoric_variance_scale={format_mean_se(result.aleatoric_variance_scale)} "
        f"fit_s={format_mean_se(result.fit_s, fmt='.3f')}"
    )


def format_draw_method_summary_aggregate(method: DrawMethodAggregate) -> str:
    return (
        f"{method.method} "
        f"avg_likelihood={format_mean_se(method.avg_likelihood)} "
        f"argmin_rms={format_mean_se(method.argmin_rms)} "
        f"argmin_hit_rate={format_mean_se(method.argmin_hit_rate, fmt='0.4f')} "
        f"eval_s={format_mean_se(method.eval_s, fmt='.3f')}"
    )


@dataclass(frozen=True)
class TurboEnnRoundResult:
    n: int
    iter_s: float
    ask_s: float
    tell_s: float


def turbo_enn_ask_stops(num_obs: int, num_ask: int) -> tuple[int, ...]:
    """Return NUM_ASK exponentially spaced cumulative observation stops in [1, NUM_OBS].

    Example: ``turbo_enn_ask_stops(30, 4) == (1, 3, 10, 30)``.
    """
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if num_ask < 1:
        raise ValueError("num_ask must be >= 1")
    if num_ask > num_obs:
        raise ValueError("num_ask must be <= num_obs")
    if num_ask == 1:
        return (num_obs,)

    ideal = np.exp(np.linspace(np.log(1.0), np.log(float(num_obs)), num=num_ask))
    stops: list[int] = []
    for i, value in enumerate(ideal):
        lo = (stops[-1] + 1) if stops else 1
        hi = num_obs - (num_ask - i - 1)
        assert lo <= hi, f"no room for stop {i}: lo={lo} hi={hi}"
        cand = int(round(float(value)))
        cand = min(max(cand, lo), hi)
        stops.append(cand)
    assert len(stops) == num_ask, f"expected {num_ask} stops, got {len(stops)}"
    assert stops[-1] == num_obs, (
        f"final stop must be num_obs={num_obs}, got {stops[-1]}"
    )
    assert all(stops[i] < stops[i + 1] for i in range(num_ask - 1)), stops
    return tuple(stops)


def build_turbo_enn_optimizer_config(
    *,
    index_driver: ENNIndexDriver,
    work_dir: str | None = None,
    num_init: int = TURBO_ENN_NUM_INIT,
):
    """Build turbo_enn config matching compare's single-metric Ackley path.

    Overrides only ``num_init`` (fixed at ``TURBO_ENN_NUM_INIT`` for the stress CLI).
    For ``BPANN_DISK``, both ``enn_storage="disk"`` and ``work_dir`` are required.
    """
    from enn.turbo.config import (
        AcqType,
        ENNFitConfig,
        ENNSurrogateConfig,
        TurboTRConfig,
        turbo_enn_config,
    )

    if num_init < 1:
        raise ValueError("num_init must be >= 1")
    enn_kwargs: dict[str, object] = {
        "k": TURBO_ENN_K,
        "fit": ENNFitConfig(num_fit_samples=TURBO_ENN_NUM_FIT_SAMPLES),
        "index_driver": index_driver,
    }
    if index_driver == ENNIndexDriver.BPANN_DISK:
        if work_dir is None:
            raise ValueError("bpann_disk requires work_dir")
        enn_kwargs["enn_storage"] = "disk"
        enn_kwargs["work_dir"] = work_dir
    elif work_dir is not None:
        raise ValueError("work_dir requires bpann_disk")
    return turbo_enn_config(
        enn=ENNSurrogateConfig(**enn_kwargs),
        trust_region=TurboTRConfig(noise_aware=True),
        acq_type=AcqType.UCB,
        num_init=num_init,
    )


def format_turbo_enn_config_header(
    *,
    num_dim: int,
    num_obs: int,
    num_ask: int,
    index_type: str,
    work_dir: str | None = None,
) -> str:
    header = (
        f"num_dim={num_dim} num_obs={num_obs} num_ask={num_ask} index_type={index_type}"
    )
    if work_dir is not None:
        header = f"{header} work_dir={work_dir}"
    return header


def format_turbo_enn_row(
    n: int, iter_s: float, ask_s: float, tell_s: float, *, n_width: int
) -> str:
    return f"{n:>{n_width}} {iter_s:.3f} {ask_s:.3f} {tell_s:.3f}"


def run_turbo_enn_stress(
    *,
    index_driver: ENNIndexDriver,
    num_dim: int,
    num_obs: int,
    num_ask: int,
    work_dir: str | None = None,
    seed: int = TURBO_ENN_SEED,
    seed_chunk: int | None = None,
    optimizer=None,
    objective=None,
    bounds: np.ndarray | None = None,
) -> Iterator[TurboEnnRoundResult]:
    """Tell DGP obs in exponential groups; at each stop time ask(1) and discard arms."""
    if num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    stops = turbo_enn_ask_stops(num_obs, num_ask)
    if seed_chunk is None:
        seed_chunk = turbo_enn_default_seed_chunk(index_driver)
    if seed_chunk < 1:
        raise ValueError("seed_chunk must be >= 1")

    from enn import create_optimizer
    from enn.benchmarks import Ackley

    if objective is None:
        ackley = Ackley(noise=TURBO_ENN_ACKLEY_NOISE, rng=np.random.default_rng(seed))
    else:
        ackley = objective
    if bounds is None:
        bounds_arr = np.array([ackley.bounds] * num_dim, dtype=float)
    else:
        bounds_arr = np.asarray(bounds, dtype=float)
    if optimizer is None:
        config = build_turbo_enn_optimizer_config(
            index_driver=index_driver,
            work_dir=work_dir,
            num_init=TURBO_ENN_NUM_INIT,
        )
        opt = create_optimizer(
            bounds=bounds_arr, config=config, rng=np.random.default_rng(seed)
        )
    else:
        opt = optimizer

    dgp_rng = np.random.default_rng(seed + 17)
    prev = 0
    for stop in stops:
        gap = stop - prev
        assert gap >= 1, f"empty group at stop={stop} prev={prev}"
        t0 = time.perf_counter()
        seed_turbo_enn_to_n(
            opt,
            ackley,
            bounds_arr,
            gap,
            rng=dgp_rng,
            chunk=seed_chunk,
        )
        tell_s = time.perf_counter() - t0
        t_ask0 = time.perf_counter()
        _ = opt.ask(num_arms=1)  # discard arms; timing only
        ask_s = time.perf_counter() - t_ask0
        iter_s = ask_s + tell_s
        yield TurboEnnRoundResult(n=stop, iter_s=iter_s, ask_s=ask_s, tell_s=tell_s)
        prev = stop


@dataclass(frozen=True)
class ProposalScaleResult:
    n: int
    ask_s: float
    tell_s: float
    proposal_s: float


def proposal_scale_ns(max_n: int) -> tuple[int, ...]:
    """Return fixed proposal-scale grid filtered by ``max_n`` (all N >= num_init)."""
    if max_n < TURBO_ENN_NUM_INIT:
        raise ValueError(f"max_n must be >= {TURBO_ENN_NUM_INIT} (TuRBO-ENN num_init)")
    return tuple(n for n in PROPOSAL_SCALE_NS if n <= max_n)


def seed_turbo_enn_to_n(
    opt,
    objective,
    bounds: np.ndarray,
    n: int,
    *,
    rng: np.random.Generator,
    chunk: int = PROPOSAL_SCALE_SEED_CHUNK,
) -> None:
    """Bulk-seed optimizer with N synthetic Ackley points via chunked ``tell``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    bounds = np.asarray(bounds, dtype=float)
    assert bounds.ndim == 2 and bounds.shape[1] == 2, (
        f"expected bounds shape (d, 2), got {bounds.shape}"
    )
    num_dim = int(bounds.shape[0])
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    assert np.all(hi > lo), "bounds require hi > lo per dimension"
    # Generate per chunk so peak RAM stays O(chunk·D), not O(n·D). Full-N
    # materialization made turbo-enn bpann_disk exceed 1 GiB well below n=1e8.
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        take = end - start
        x = rng.uniform(lo, hi, size=(take, num_dim))
        y = np.asarray(objective(x), dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        assert y.shape == (take, 1), f"expected y shape ({take}, 1), got {y.shape}"
        opt.tell(x, y)


def _timed_ask_tell_round(opt, objective) -> tuple[float, float]:
    """Run one ask → eval → tell round; return (ask_s, tell_s)."""
    t0 = time.perf_counter()
    x = opt.ask(num_arms=1)
    ask_s = time.perf_counter() - t0
    y = np.asarray(objective(x), dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    assert y.shape == (1, 1), f"expected y shape (1, 1), got {y.shape}"
    t_tell0 = time.perf_counter()
    opt.tell(x, y)
    tell_s = time.perf_counter() - t_tell0
    return ask_s, tell_s


def probe_turbo_enn_proposal(
    opt,
    objective,
    *,
    warmup: int = PROPOSAL_SCALE_WARMUP,
    num_probes: int = PROPOSAL_SCALE_NUM_PROBES,
) -> tuple[float, float, float]:
    """Warmup then time ask/tell probes; return mean ask_s, tell_s, proposal_s."""
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if num_probes < 1:
        raise ValueError("num_probes must be >= 1")

    for _ in range(warmup):
        _timed_ask_tell_round(opt, objective)

    ask_times: list[float] = []
    tell_times: list[float] = []
    for _ in range(num_probes):
        ask_s, tell_s = _timed_ask_tell_round(opt, objective)
        ask_times.append(ask_s)
        tell_times.append(tell_s)

    ask_mean = float(np.mean(ask_times))
    tell_mean = float(np.mean(tell_times))
    proposal_mean = ask_mean + tell_mean
    return ask_mean, tell_mean, proposal_mean


def format_proposal_scale_config_header(
    *,
    num_dim: int,
    max_n: int,
    num_probes: int,
    index_type: str,
    work_dir: str | None = None,
) -> str:
    header = (
        f"num_dim={num_dim} max_n={max_n} num_probes={num_probes} "
        f"index_type={index_type}"
    )
    if work_dir is not None:
        header = f"{header} work_dir={work_dir}"
    return header


def format_proposal_scale_row(
    n: int, ask_s: float, tell_s: float, proposal_s: float, *, n_width: int
) -> str:
    return f"{n:>{n_width}} {ask_s:.3f} {tell_s:.3f} {proposal_s:.3f}"


def run_proposal_scale_stress(
    index_driver: ENNIndexDriver,
    num_dim: int,
    max_n: int,
    *,
    work_dir: str | None = None,
    num_probes: int = PROPOSAL_SCALE_NUM_PROBES,
    warmup: int = PROPOSAL_SCALE_WARMUP,
    seed_chunk: int = PROPOSAL_SCALE_SEED_CHUNK,
    seed: int = TURBO_ENN_SEED,
    optimizer_factory=None,
    objective=None,
) -> Iterator[ProposalScaleResult]:
    """Fresh opt per N: seed to N, warmup, probe; yield mean proposal timings."""
    if num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    if num_probes < 1:
        raise ValueError("num_probes must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if seed_chunk < 1:
        raise ValueError("seed_chunk must be >= 1")
    grid = proposal_scale_ns(max_n)

    from enn import create_optimizer
    from enn.benchmarks import Ackley

    if objective is None:
        ackley = Ackley(noise=TURBO_ENN_ACKLEY_NOISE, rng=np.random.default_rng(seed))
    else:
        ackley = objective
    bounds = np.array([ackley.bounds] * num_dim, dtype=float)

    for n in grid:
        n_work_dir: str | None
        if work_dir is not None:
            n_work_dir = str(Path(work_dir) / f"n{n}")
            Path(n_work_dir).mkdir(parents=True, exist_ok=True)
        else:
            n_work_dir = None
        if optimizer_factory is not None:
            opt = optimizer_factory(n=n, work_dir=n_work_dir)
        else:
            config = build_turbo_enn_optimizer_config(
                index_driver=index_driver,
                work_dir=n_work_dir,
                num_init=TURBO_ENN_NUM_INIT,
            )
            opt = create_optimizer(
                bounds=bounds, config=config, rng=np.random.default_rng(seed + n)
            )
        seed_turbo_enn_to_n(
            opt,
            ackley,
            bounds,
            n,
            rng=np.random.default_rng(seed + 10_000 + n),
            chunk=seed_chunk,
        )
        ask_mean, tell_mean, proposal_mean = probe_turbo_enn_proposal(
            opt,
            ackley,
            warmup=warmup,
            num_probes=num_probes,
        )
        yield ProposalScaleResult(
            n=n, ask_s=ask_mean, tell_s=tell_mean, proposal_s=proposal_mean
        )


@click.group()
def cli() -> None:
    """Operational stress tools."""


@cli.command(
    "enn",
    params=[
        click.Argument(
            ["index_type"],
            type=click.Choice(INDEX_TYPE_CHOICES),
        ),
        click.Argument(["num_obs"], type=int),
        click.Option(
            ["--num-dim"],
            type=int,
            default=DEFAULT_NUM_DIM,
            show_default=True,
            help="Embedding dimension for synthetic observations.",
        ),
        click.Option(
            ["--progress-every"],
            type=int,
            default=0,
            show_default=True,
            help="Emit `progress n=<N>` to stderr every N additions (0 disables).",
        ),
        click.Option(
            ["--heartbeat-seconds"],
            type=float,
            default=DEFAULT_HEARTBEAT_SECONDS,
            show_default=True,
            help="Emit `heartbeat n=<N>` to stderr at most this often (0 disables).",
        ),
        click.Option(
            ["--work-dir"],
            type=click.Path(file_okay=False, dir_okay=True, path_type=str),
            default=None,
            help="Disk-backed ENN work directory (requires bpann_disk).",
        ),
        click.Option(
            ["--batch"],
            is_flag=True,
            default=False,
            help=(
                "Add observations in batches of "
                f"{ENN_ADD_STRESS_BATCH_SIZE} instead of one row per add."
            ),
        ),
        click.Option(
            ["--ennbo-config"],
            type=click.Path(file_okay=True, dir_okay=False, path_type=str),
            default=None,
            help="Path to ennbo config.toml (overrides ~/.ennbo/config.toml).",
        ),
    ],
)
def enn(
    index_type: str,
    num_obs: int,
    num_dim: int,
    progress_every: int,
    heartbeat_seconds: float,
    work_dir: str | None,
    batch: bool,
    ennbo_config: str | None,
) -> None:
    """Time 1000-point ENN queries at sparse checkpoints while streaming adds."""
    if ennbo_config is not None:
        from enn._rust import set_config_path

        set_config_path(ennbo_config)
    if num_obs < 1:
        raise click.ClickException("num_obs must be >= 1")
    if num_dim < 1:
        raise click.ClickException("num_dim must be >= 1")
    if progress_every < 0:
        raise click.ClickException("progress_every must be >= 0")
    if heartbeat_seconds < 0:
        raise click.ClickException("heartbeat_seconds must be >= 0")
    if work_dir is not None and index_type not in DISK_INDEX_TYPE_CHOICES:
        raise click.ClickException(
            f"work_dir requires index_type in {sorted(DISK_INDEX_TYPE_CHOICES)}"
        )
    if index_type in DISK_INDEX_TYPE_CHOICES and work_dir is None:
        raise click.ClickException(f"{index_type} requires --work-dir")
    driver = parse_index_driver(index_type)
    num_obs_existing = load_num_obs_existing(work_dir) if work_dir is not None else 0
    click.echo(
        format_config_header(
            num_dim=num_dim,
            num_obs=num_obs,
            work_dir=work_dir,
            num_obs_existing=num_obs_existing if num_obs_existing else None,
        )
    )
    n_width = stress_row_n_width(num_obs)
    batch_size = ENN_ADD_STRESS_BATCH_SIZE if batch else 1
    for n, query_s, segment_s in run_enn_add_stress(
        index_driver=driver,
        num_obs=num_obs,
        config=EnnAddStressConfig(
            num_dim=num_dim,
            batch_size=batch_size,
            progress_every=progress_every,
            heartbeat_seconds=heartbeat_seconds,
            work_dir=work_dir,
        ),
    ):
        click.echo(format_stress_row(n, query_s, segment_s, n_width=n_width))


@cli.command(
    "sample",
    params=[
        click.Argument(
            ["work_dir"],
            type=click.Path(file_okay=False, dir_okay=True, path_type=str),
        ),
        click.Argument(["num_samples"], type=int),
        click.Option(
            ["--seed"],
            type=int,
            default=STRESS_QUERY_SEED,
            show_default=True,
            help="RNG seed for query points and function draw.",
        ),
    ],
)
def sample(work_dir: str, num_samples: int, seed: int) -> None:
    """Draw posterior function samples at uniform random x on a persisted bpann store."""
    if num_samples < 1:
        raise click.ClickException("num_samples must be >= 1")
    if not Path(work_dir).is_dir():
        raise click.ClickException(f"work_dir does not exist: {work_dir}")
    try:
        result = run_sample_stress(
            work_dir=work_dir,
            config=SampleStressConfig(num_samples=num_samples, seed=seed),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not result.all_finite:
        raise click.ClickException("posterior_function_draw returned non-finite values")
    click.echo(format_sample_config_header(result=result, work_dir=work_dir))
    click.echo(format_sample_summary(result))


@cli.command(
    "draw",
    params=[
        click.Argument(["num_obs"], type=int),
        click.Argument(["num_test"], type=int),
        click.Option(
            ["--num-dim"],
            type=int,
            default=DEFAULT_NUM_DIM,
            show_default=True,
            help="Embedding dimension for synthetic observations.",
        ),
        click.Option(
            ["--seed"],
            type=int,
            default=DEFAULT_DRAW_SEED,
            show_default=True,
            help="RNG seed for train/test data (fit uses seed+1).",
        ),
        click.Option(
            ["--k"],
            type=int,
            default=DEFAULT_DRAW_K,
            show_default=True,
            help="Number of neighbors for fit and posterior.",
        ),
        click.Option(
            ["--num-fit-candidates"],
            type=int,
            default=DEFAULT_DRAW_NUM_FIT_CANDIDATES,
            show_default=True,
            help="Hyperparameter candidates per fit ask.",
        ),
        click.Option(
            ["--num-fit-samples"],
            type=int,
            default=DEFAULT_DRAW_NUM_FIT_SAMPLES,
            show_default=True,
            help="Subsample size for fit log-likelihood.",
        ),
        click.Option(
            ["--num-draws"],
            type=int,
            default=DEFAULT_DRAW_NUM_DRAWS,
            show_default=True,
            help="Number of posterior draws for likelihood and argmin metrics.",
        ),
        click.Option(
            ["--num-seeds"],
            type=int,
            default=DEFAULT_DRAW_NUM_SEEDS,
            show_default=True,
            help="Repeat full draw stress over seed..seed+num_seeds-1; report mean ± SE.",
        ),
    ],
)
def draw(
    num_obs: int,
    num_test: int,
    num_dim: int,
    seed: int,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int,
    num_draws: int,
    num_seeds: int,
) -> None:
    """Fit ENN on synthetic data; report avg likelihood for two draw methods."""
    if num_obs < 1:
        raise click.ClickException("num_obs must be >= 1")
    if num_test < 1:
        raise click.ClickException("num_test must be >= 1")
    if num_dim < 1:
        raise click.ClickException("num_dim must be >= 1")
    if k < 1:
        raise click.ClickException("k must be >= 1")
    if num_fit_candidates < 1:
        raise click.ClickException("num_fit_candidates must be >= 1")
    if num_fit_samples < 1:
        raise click.ClickException("num_fit_samples must be >= 1")
    if num_draws < 2:
        raise click.ClickException("num_draws must be >= 2")
    if num_seeds < 1:
        raise click.ClickException("num_seeds must be >= 1")
    config = DrawStressConfig(
        num_obs=num_obs,
        num_test=num_test,
        num_dim=num_dim,
        seed=seed,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        num_draws=num_draws,
    )
    try:
        agg = run_draw_stress_over_seeds(config, num_seeds=num_seeds)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(format_draw_config_header_aggregate(agg))
    click.echo(format_draw_method_summary_aggregate(agg.posterior))
    click.echo(format_draw_method_summary_aggregate(agg.posterior_function_draw))


@cli.command(
    "turbo-enn",
    params=[
        click.Argument(
            ["index_type"],
            type=click.Choice(INDEX_TYPE_CHOICES),
        ),
        click.Argument(["num_obs"], type=int),
        click.Argument(["num_ask"], type=int),
        click.Option(
            ["--num-dim"],
            type=int,
            default=DEFAULT_NUM_DIM,
            show_default=True,
            help="Embedding dimension for Ackley / TuRBO-ENN.",
        ),
        click.Option(
            ["--work-dir"],
            type=click.Path(file_okay=False, dir_okay=True, path_type=str),
            default=None,
            help="Disk-backed ENN work directory (requires bpann_disk).",
        ),
    ],
)
def turbo_enn(
    index_type: str,
    num_obs: int,
    num_ask: int,
    num_dim: int,
    work_dir: str | None,
) -> None:
    """Time TuRBO-ENN ask at exponentially spaced DGP observation checkpoints."""
    if num_dim < 1:
        raise click.ClickException("num_dim must be >= 1")
    if num_obs < 1:
        raise click.ClickException("num_obs must be >= 1")
    if num_ask < 1:
        raise click.ClickException("num_ask must be >= 1")
    if num_ask > num_obs:
        raise click.ClickException("num_ask must be <= num_obs")
    if work_dir is not None and index_type not in DISK_INDEX_TYPE_CHOICES:
        raise click.ClickException(
            f"work_dir requires index_type in {sorted(DISK_INDEX_TYPE_CHOICES)}"
        )
    if index_type in DISK_INDEX_TYPE_CHOICES and work_dir is None:
        raise click.ClickException(f"{index_type} requires --work-dir")
    driver = parse_index_driver(index_type)
    click.echo(
        format_turbo_enn_config_header(
            num_dim=num_dim,
            num_obs=num_obs,
            num_ask=num_ask,
            index_type=index_type,
            work_dir=work_dir,
        )
    )
    n_width = len(str(num_obs))
    try:
        for row in run_turbo_enn_stress(
            index_driver=driver,
            num_dim=num_dim,
            num_obs=num_obs,
            num_ask=num_ask,
            work_dir=work_dir,
        ):
            click.echo(
                format_turbo_enn_row(
                    row.n,
                    row.iter_s,
                    row.ask_s,
                    row.tell_s,
                    n_width=n_width,
                )
            )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command(
    "proposal-scale",
    params=[
        click.Argument(
            ["index_type"],
            type=click.Choice(INDEX_TYPE_CHOICES),
        ),
        click.Option(
            ["--num-dim"],
            type=int,
            default=DEFAULT_NUM_DIM,
            show_default=True,
            help="Embedding dimension for Ackley / TuRBO-ENN.",
        ),
        click.Option(
            ["--max-n"],
            type=int,
            default=PROPOSAL_SCALE_NS[-1],
            show_default=True,
            help="Largest N on the proposal-scale grid (filters fixed checkpoints).",
        ),
        click.Option(
            ["--num-probes"],
            type=int,
            default=PROPOSAL_SCALE_NUM_PROBES,
            show_default=True,
            help="Timed ask/tell rounds per N after warmup.",
        ),
        click.Option(
            ["--work-dir"],
            type=click.Path(file_okay=False, dir_okay=True, path_type=str),
            default=None,
            help="Disk-backed ENN work directory (requires bpann_disk).",
        ),
    ],
)
def proposal_scale(
    index_type: str,
    num_dim: int,
    max_n: int,
    num_probes: int,
    work_dir: str | None,
) -> None:
    """Time mean TuRBO-ENN proposal cost at log-spaced N (seed then probe)."""
    if num_dim < 1:
        raise click.ClickException("num_dim must be >= 1")
    if max_n < TURBO_ENN_NUM_INIT:
        raise click.ClickException(
            f"max_n must be >= {TURBO_ENN_NUM_INIT} (TuRBO-ENN num_init)"
        )
    if num_probes < 1:
        raise click.ClickException("num_probes must be >= 1")
    if work_dir is not None and index_type not in DISK_INDEX_TYPE_CHOICES:
        raise click.ClickException(
            f"work_dir requires index_type in {sorted(DISK_INDEX_TYPE_CHOICES)}"
        )
    if index_type in DISK_INDEX_TYPE_CHOICES and work_dir is None:
        raise click.ClickException(f"{index_type} requires --work-dir")
    driver = parse_index_driver(index_type)
    click.echo(
        format_proposal_scale_config_header(
            num_dim=num_dim,
            max_n=max_n,
            num_probes=num_probes,
            index_type=index_type,
            work_dir=work_dir,
        )
    )
    n_width = len(str(max_n))
    try:
        for row in run_proposal_scale_stress(
            driver,
            num_dim,
            max_n,
            work_dir=work_dir,
            num_probes=num_probes,
        ):
            click.echo(
                format_proposal_scale_row(
                    row.n,
                    row.ask_s,
                    row.tell_s,
                    row.proposal_s,
                    n_width=n_width,
                )
            )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
