from __future__ import annotations

import time

import numpy as np

from enn.ennts import ENNTS, ENNTSConfig
from evals.stress_eval import format_plain, format_smaller


def _model(dim, seed) -> ENNTS:
    lookback = 5
    return ENNTS(
        ENNTSConfig(
            lookback=lookback,
            num_tables=4,
            num_projections=3,
            bucket_width=max(1.0, float(np.sqrt(lookback * dim))),
            k=4,
            probe=0,
            seed=int(seed),
        )
    )


def _fill(model, n, dim, rng) -> None:
    model.add(0, rng.normal(size=(n, dim)), rng.normal(size=(n, 1)))


def time_cell(n, batch, dim, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed + 17 * n + 31 * batch + dim)
    model = _model(dim, seed)
    _fill(model, n, dim, rng)
    query = rng.normal(size=(batch, 5, dim))
    started = time.perf_counter()
    model.search(query, cutoff=n + 5, require_target=False)
    query_s = time.perf_counter() - started
    started = time.perf_counter()
    model.add(0, rng.normal(size=(batch, dim)), rng.normal(size=(batch, 1)))
    add_s = time.perf_counter() - started
    return add_s, query_s


def format_timing_line(n, batch, dim, add_s, query_s) -> str:
    return (
        "EVAL: "
        f"{format_plain('n', n)} "
        f"{format_plain('b', batch)} "
        f"{format_plain('d', dim)} "
        f"{format_smaller('add_s', f'{add_s:.6f}')} "
        f"{format_smaller('query_s', f'{query_s:.6f}')}"
    )


def run_ennts_timing_eval() -> None:
    for n in (64, 256):
        for batch in (1, 16):
            for dim in (2, 8):
                add_s, query_s = time_cell(n, batch, dim)
                print(format_timing_line(n, batch, dim, add_s, query_s), flush=True)
