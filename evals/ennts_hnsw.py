from __future__ import annotations

import numpy as np

from enn.ennts import ENNTS, ENNTSConfig
from evals.ennts_roll import (
    build_series,
    format_stress_line,
    roll_spec,
    rolling,
)
from evals.stress_eval import format_plain

N_SERIES = 12
LENGTH = 260
DIM = 24
LOOKBACK = 8
HORIZON = 1
CADENCE = 1
BURN = LOOKBACK + HORIZON + 12
SEEDS = (0, 1)


def make_wide_series(rng, _series_id):
    noise = rng.normal(scale=0.25, size=(LENGTH, DIM))
    walk = np.cumsum(rng.normal(scale=0.55, size=LENGTH))
    noise[:, 0] = walk
    totals = np.cumsum(walk)
    prev = np.concatenate([np.zeros(LOOKBACK), totals[:-LOOKBACK]])
    mean = (totals - prev) / LOOKBACK
    y = np.zeros((LENGTH, 1))
    last = LENGTH - HORIZON
    y[HORIZON + LOOKBACK - 1 :, 0] = np.tanh(mean[LOOKBACK - 1 : last])
    y[:, 0] += rng.normal(scale=0.05, size=LENGTH)
    return noise, y


def hnsw_model(seed):
    return ENNTS(
        ENNTSConfig(
            lookback=LOOKBACK,
            horizon=HORIZON,
            cadence=CADENCE,
            num_tables=6,
            num_projections=4,
            bucket_width=4.0,
            k=8,
            probe=1,
            seed=int(seed),
            aleatoric_variance_scale=0.0025,
        )
    )


def run_ennts_hnsw_eval() -> None:
    spec = roll_spec(LENGTH, CADENCE, LOOKBACK, HORIZON, BURN, False)
    for seed in SEEDS:
        xs, ys = build_series(make_wide_series, N_SERIES, seed)
        scores = rolling(hnsw_model(seed), xs, ys, spec)
        extra = [format_plain("windows", scores["windows"])]
        print(format_stress_line("hnsw", seed, scores, extra), flush=True)
