#!/usr/bin/env python

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import click
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from enn.turbo.config.enn_index_driver import ENNIndexDriver

INDEX_TYPE_CHOICES: tuple[str, ...] = ("flat", "hnsw", "hnsw_usearch")
DEFAULT_NUM_DIM = 10
STRESS_OBS_BATCH_SIZE = 100
DEFAULT_HEARTBEAT_SECONDS = 10.0
STRESS_QUERY_N = 1000
STRESS_QUERY_SEED = 1
STRESS_QUERY_K = 10
STRESS_PARAMS = ENNParams(
    k_num_neighbors=STRESS_QUERY_K,
    epistemic_variance_scale=1.0,
    aleatoric_variance_scale=0.1,
)


@dataclass(frozen=True)
class EnnAddStressConfig:
    num_dim: int = DEFAULT_NUM_DIM
    seed: int = 0
    progress_every: int = 0
    heartbeat_seconds: float = 0.0
    query_n: int = STRESS_QUERY_N
    query_seed: int = STRESS_QUERY_SEED
    index_path: str | None = None


def parse_index_driver(name: str) -> ENNIndexDriver:
    mapping = {
        "flat": ENNIndexDriver.FLAT,
        "hnsw": ENNIndexDriver.HNSW,
        "hnsw_usearch": ENNIndexDriver.HNSW_USEARCH,
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


def format_config_header(
    *, num_dim: int, num_obs: int, index_path: str | None = None
) -> str:
    header = f"num_dim={num_dim} num_obs={num_obs}"
    if index_path is not None:
        header = f"{header} index_path={index_path}"
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
) -> Iterator[tuple[int, float]]:
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
    if cfg.index_path is not None:
        model_kwargs["index_path"] = cfg.index_path
    model = EpistemicNearestNeighbors(**model_kwargs)

    last_heartbeat_t = time.perf_counter()
    for n, (x_row, y_row) in enumerate(
        iter_synthetic_observations(num_obs, num_dim=cfg.num_dim, seed=cfg.seed),
        start=1,
    ):
        model.add(x_row, y_row)
        if cfg.progress_every and (n % cfg.progress_every == 0):
            click.echo(f"progress n={n}", err=True)
        if cfg.heartbeat_seconds and (
            time.perf_counter() - last_heartbeat_t >= cfg.heartbeat_seconds
        ):
            click.echo(f"heartbeat n={n}", err=True)
            last_heartbeat_t = time.perf_counter()
        if n in checkpoints:
            model.sync_index()
            query_s = _time_query_s(model, x_query)
            yield (n, query_s)


def stress_row_n_width(num_obs: int) -> int:
    """Character width for the N column; sized for the largest checkpoint (num_obs)."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    return len(str(num_obs))


def format_stress_row(n: int, query_s: float, *, n_width: int) -> str:
    return f"{n:>{n_width}} {query_s:.3f}"


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
            ["--index-path"],
            type=click.Path(),
            default=None,
            help="Optional USearch index file (requires hnsw_usearch). Persists at each checkpoint sync.",
        ),
    ],
)
def enn(
    index_type: str,
    num_obs: int,
    num_dim: int,
    progress_every: int,
    heartbeat_seconds: float,
    index_path: str | None,
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
    if index_path is not None and index_type != "hnsw_usearch":
        raise click.ClickException("index_path requires index_type hnsw_usearch")
    driver = parse_index_driver(index_type)
    click.echo(
        format_config_header(num_dim=num_dim, num_obs=num_obs, index_path=index_path)
    )
    n_width = stress_row_n_width(num_obs)
    for n, query_s in run_enn_add_stress(
        index_driver=driver,
        num_obs=num_obs,
        config=EnnAddStressConfig(
            num_dim=num_dim,
            progress_every=progress_every,
            heartbeat_seconds=heartbeat_seconds,
            index_path=index_path,
        ),
    ):
        click.echo(format_stress_row(n, query_s, n_width=n_width))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
