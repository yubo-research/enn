#!/usr/bin/env python

from __future__ import annotations

import json
import resource
import struct
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from enn.turbo.config.enn_index_driver import ENNIndexDriver

INDEX_TYPE_CHOICES: tuple[str, ...] = ("flat", "bpann_disk")
DISK_INDEX_TYPE_CHOICES: frozenset[str] = frozenset({"bpann_disk"})
DISK_DEFER_SYNC_DRIVERS: frozenset[ENNIndexDriver] = frozenset(
    {ENNIndexDriver.BPANN_DISK}
)
DEFAULT_NUM_DIM = 10
STRESS_OBS_BATCH_SIZE = 100
DEFAULT_HEARTBEAT_SECONDS = 10.0
STRESS_QUERY_N = 1000
STRESS_QUERY_SEED = 1
STRESS_QUERY_K = 10
NUM_SAMPLE = STRESS_QUERY_N
DEFAULT_SAMPLE_WORK_DIR = "_enn"
DEFAULT_SAMPLE_X_LOW = -1.0
DEFAULT_SAMPLE_X_HIGH = 1.0
STRESS_PARAMS = ENNParams(
    k_num_neighbors=STRESS_QUERY_K,
    epistemic_variance_scale=1.0,
    aleatoric_variance_scale=0.1,
)
DISK_STRESS_RSS_BASELINE_MIB = 512
DISK_STRESS_RSS_PER_SHARD_ROW_BYTES = 64
DEFAULT_SHARD_MAX_ROWS = 500_000


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
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (n, num_dim) x and (n, 1) y batches without holding all num_obs rows."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    rng = np.random.default_rng(seed)
    emitted = 0
    while emitted < num_obs:
        n = min(batch_size, num_obs - emitted)
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

    last_heartbeat_t = time.perf_counter()
    last_checkpoint_t = time.perf_counter()
    try:
        for n, (x_row, y_row) in enumerate(
            iter_synthetic_observations(num_obs, num_dim=cfg.num_dim, seed=cfg.seed),
            start=1,
        ):
            model.add(x_row, y_row)
            if index_driver in DISK_DEFER_SYNC_DRIVERS:
                model.schedule_background_flush()
            if cfg.progress_every and (n % cfg.progress_every == 0):
                click.echo(f"progress n={n}", err=True)
            if cfg.heartbeat_seconds and (
                time.perf_counter() - last_heartbeat_t >= cfg.heartbeat_seconds
            ):
                click.echo(f"heartbeat n={n}", err=True)
                last_heartbeat_t = time.perf_counter()
            if n in checkpoints:
                if index_driver not in DISK_DEFER_SYNC_DRIVERS:
                    model.ensure_index_sync()
                segment_s = time.perf_counter() - last_checkpoint_t
                query_s = _time_query_s(model, x_query)
                last_checkpoint_t = time.perf_counter()
                yield (n, query_s, segment_s)
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
    ],
)
def enn(
    index_type: str,
    num_obs: int,
    num_dim: int,
    progress_every: int,
    heartbeat_seconds: float,
    work_dir: str | None,
) -> None:
    """Time 1000-point ENN queries at sparse checkpoints while streaming adds."""
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
    for n, query_s, segment_s in run_enn_add_stress(
        index_driver=driver,
        num_obs=num_obs,
        config=EnnAddStressConfig(
            num_dim=num_dim,
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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
