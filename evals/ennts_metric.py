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

N_SERIES = 6
LENGTH = 340
DIM = 8
LOOKBACK = 5
HORIZON = 1
CADENCE = 1
BURN = LOOKBACK + HORIZON + 8
SEEDS = (0, 1)


def make_scaled_series(rng, _series_id):
    signal = rng.normal(size=LENGTH)
    x = rng.normal(scale=12.0, size=(LENGTH, DIM))
    x[:, 0] = 0.015 * signal
    y = np.zeros((LENGTH, 1))
    y[HORIZON:, 0] = np.sin(signal[:-HORIZON])
    y[:, 0] += rng.normal(scale=0.08, size=LENGTH)
    return x, y


def metric_model(seed):
    return ENNTS(
        ENNTSConfig(
            lookback=LOOKBACK,
            horizon=HORIZON,
            cadence=CADENCE,
            num_tables=6,
            num_projections=3,
            bucket_width=1.0,
            k=8,
            probe=1,
            seed=int(seed),
            aleatoric_variance_scale=0.0064,
            metric_learning="auto",
        )
    )


def run_ennts_metric_eval() -> None:
    spec = roll_spec(LENGTH, CADENCE, LOOKBACK, HORIZON, BURN, True)
    for seed in SEEDS:
        xs, ys = build_series(make_scaled_series, N_SERIES, seed)
        model = metric_model(seed)
        scores = rolling(model, xs, ys, spec)
        learned = int(model.uses_learned_metric)
        extra = [
            format_plain("learned", learned),
            format_plain("morph", model.metric_morph_remaining),
        ]
        print(format_stress_line("metric", seed, scores, extra), flush=True)
