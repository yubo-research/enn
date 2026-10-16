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

N_SERIES = 14
LENGTH = 140
DIM = 6
LOOKBACK = 8
HORIZON = 12
CADENCE = 3
BURN = LOOKBACK + HORIZON + 6
SEEDS = (0, 1)


def make_delayed_series(rng, series_id):
    phase = int(series_id) * 3
    latent = np.cumsum(rng.normal(scale=0.45, size=LENGTH + phase))
    x = np.zeros((LENGTH, DIM))
    x[:, 0] = latent[phase : phase + LENGTH]
    decoy = np.zeros(LENGTH)
    decoy[1:] = np.tanh(x[:-1, 0])
    x[:, 1] = 5.0 * decoy
    x[:, 2:] = rng.normal(scale=0.15, size=(LENGTH, DIM - 2))
    y = np.zeros((LENGTH, 1))
    y[HORIZON:, 0] = np.tanh(x[:-HORIZON, 0])
    y[:, 0] += rng.normal(scale=0.04, size=LENGTH)
    return x, y


def horizon_model(seed):
    return ENNTS(
        ENNTSConfig(
            lookback=LOOKBACK,
            horizon=HORIZON,
            cadence=CADENCE,
            num_tables=8,
            num_projections=4,
            bucket_width=4.0,
            k=8,
            probe=1,
            seed=int(seed),
            aleatoric_variance_scale=0.0016,
        )
    )


def run_ennts_horizon_eval() -> None:
    spec = roll_spec(LENGTH, CADENCE, LOOKBACK, HORIZON, BURN, False)
    for seed in SEEDS:
        xs, ys = build_series(make_delayed_series, N_SERIES, seed)
        scores = rolling(horizon_model(seed), xs, ys, spec)
        extra = [
            format_plain("horizon", HORIZON),
            format_plain("cadence", CADENCE),
        ]
        print(format_stress_line("horizon", seed, scores, extra), flush=True)
