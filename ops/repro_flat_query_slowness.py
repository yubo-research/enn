#!/usr/bin/env python3
"""Repro: ENN FLAT (Rust ``exact``) neighbor queries are slow at scale.

FLAT maps to Rust index driver name ``exact``
(``ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.FLAT] == "exact"``).

This script builds FLAT and BPANN_DISK models on the same random data, syncs
indexes, then times a batched ``posterior()`` call. Hyperparameter fitting is
skipped so wall time isolates index query cost.

Measured on one host (N=100_000, Q=10_000, D=100, k=9), before Flat Faiss
SIMD+OpenMP tuning:

    FLAT       query_sec ≈ 95.6
    BPANN_DISK query_sec ≈ 9.1
    ratio                ≈ 10.5×

After preferring Faiss SIMD distances and enabling OpenMP for Flat search:

    FLAT       query_sec ≈ 2.7
    (mid-scale N=20_000 / Q=2_000: ≈ 0.11 vs prior ≈ 3.8)

At N=20_000 / Q=2_000 / D=100 the same pattern appeared before the fix
(~4.0s vs ~0.4s).

Example
-------
    python ops/repro_flat_query_slowness.py
    python ops/repro_flat_query_slowness.py --num-obs 20000 --num-query 2000
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass

import click
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver, ENN_INDEX_DRIVER_TO_RUST

_DEFAULT_NUM_OBS = 100_000
_DEFAULT_NUM_QUERY = 10_000
_DEFAULT_NUM_DIM = 100
_DEFAULT_K = 9
_DEFAULT_SEED = 17
_ADD_BATCH = 1_000


@dataclass(frozen=True)
class TimedBuild:
    model: EpistemicNearestNeighbors
    add_sec: float
    sync_sec: float


def _make_data(num_obs: int, num_query: int, num_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    train_x = rng.standard_normal((num_obs, num_dim))
    # Simple synthetic response; posterior timing does not depend on y values.
    train_y = (train_x[:, : min(3, num_dim)].sum(axis=1, keepdims=True)).astype(float)
    train_yvar = 0.01 * np.ones_like(train_y)
    query_x = rng.standard_normal((num_query, num_dim))
    return train_x, train_y, train_yvar, query_x


def _empty_model(num_dim: int, *, driver: ENNIndexDriver, work_dir: str | None):
    empty_x = np.empty((0, num_dim))
    empty_y = np.empty((0, 1))
    empty_yvar = np.empty((0, 1))
    kwargs: dict = {"index_driver": driver}
    if driver == ENNIndexDriver.BPANN_DISK:
        kwargs["enn_storage"] = "disk"
        kwargs["work_dir"] = work_dir
    return EpistemicNearestNeighbors(empty_x, empty_y, empty_yvar, **kwargs)


def _build(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_yvar: np.ndarray,
    *,
    driver: ENNIndexDriver,
    work_dir: str | None,
    batch_size: int,
) -> TimedBuild:
    model = _empty_model(int(train_x.shape[1]), driver=driver, work_dir=work_dir)
    n = int(train_x.shape[0])
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        model.add(train_x[start:end], train_y[start:end], train_yvar[start:end])
    add_sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    model.ensure_index_sync()
    sync_sec = time.perf_counter() - t1
    return TimedBuild(model=model, add_sec=add_sec, sync_sec=sync_sec)


def _time_posterior(model: EpistemicNearestNeighbors, query_x: np.ndarray, params: ENNParams) -> float:
    flags = PosteriorFlags(observation_noise=True)
    t0 = time.perf_counter()
    _ = model.posterior(query_x, params=params, flags=flags)
    return time.perf_counter() - t0


def _print_row(label: str, rust_name: str, build: TimedBuild, query_sec: float) -> None:
    click.echo(
        f"{label:12s} rust={rust_name:10s} "
        f"add_sec={build.add_sec:8.4f} sync_sec={build.sync_sec:8.4f} "
        f"query_sec={query_sec:8.4f}"
    )


@click.command()
@click.option("--num-obs", type=int, default=_DEFAULT_NUM_OBS, show_default=True)
@click.option("--num-query", type=int, default=_DEFAULT_NUM_QUERY, show_default=True)
@click.option("--num-dim", type=int, default=_DEFAULT_NUM_DIM, show_default=True)
@click.option("--k", "k_neighbors", type=int, default=_DEFAULT_K, show_default=True)
@click.option("--seed", type=int, default=_DEFAULT_SEED, show_default=True)
@click.option(
    "--batch-size",
    type=int,
    default=_ADD_BATCH,
    show_default=True,
    help="Observations per add() call before ensure_index_sync().",
)
@click.option(
    "--skip-bpann/--with-bpann",
    default=False,
    show_default=True,
    help="Skip BPANN_DISK comparison (FLAT-only timing).",
)
def main(
    num_obs: int,
    num_query: int,
    num_dim: int,
    k_neighbors: int,
    seed: int,
    batch_size: int,
    skip_bpann: bool,
) -> None:
    """Time ENN FLAT/exact vs BPANN_DISK posterior queries."""
    if num_obs < k_neighbors:
        raise click.ClickException(f"--num-obs must be >= k ({k_neighbors})")
    if num_query < 1:
        raise click.ClickException("--num-query must be >= 1")

    train_x, train_y, train_yvar, query_x = _make_data(
        num_obs, num_query, num_dim, seed
    )
    params = ENNParams(
        k_num_neighbors=k_neighbors,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
    )

    click.echo(
        f"repro_flat_query_slowness: N={num_obs} Q={num_query} D={num_dim} "
        f"k={k_neighbors} seed={seed}"
    )
    click.echo(
        f"FLAT rust name = {ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.FLAT]!r} "
        f"(this is the exact/brute-force path)"
    )

    flat = _build(
        train_x,
        train_y,
        train_yvar,
        driver=ENNIndexDriver.FLAT,
        work_dir=None,
        batch_size=batch_size,
    )
    flat_query = _time_posterior(flat.model, query_x, params)
    _print_row("FLAT", ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.FLAT], flat, flat_query)

    if skip_bpann:
        click.echo(
            "Skipped BPANN_DISK. FLAT query_sec above is the slow exact path."
        )
        return

    with tempfile.TemporaryDirectory(prefix="enn_repro_bpann_") as work_dir:
        bpann = _build(
            train_x,
            train_y,
            train_yvar,
            driver=ENNIndexDriver.BPANN_DISK,
            work_dir=work_dir,
            batch_size=batch_size,
        )
        bpann_query = _time_posterior(bpann.model, query_x, params)
        _print_row(
            "BPANN_DISK",
            ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.BPANN_DISK],
            bpann,
            bpann_query,
        )

    if bpann_query > 0:
        ratio = flat_query / bpann_query
        click.echo(f"query_sec ratio FLAT/BPANN_DISK = {ratio:0.1f}x")
    click.echo(
        "Result: FLAT/exact posterior queries dominate wall time at this scale "
        f"(FLAT query_sec={flat_query:0.4f}, BPANN query_sec={bpann_query:0.4f})."
    )


if __name__ == "__main__":
    main()
