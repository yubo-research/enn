
from __future__ import annotations

import numpy as np

from evals.metric_12d import MODELS, NUM_DIM, Metric12dConfig, run_eval as run_stream_eval
from evals.metric_12d_results import CheckpointSummary

NOISE_STD = 0.1
RANGES = 10.0 ** np.linspace(-2.0, 2.0, NUM_DIM)


def make_data(num_obs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.random((num_obs, NUM_DIM))
    y = (u - 0.5).sum(axis=1) + NOISE_STD * rng.standard_normal(num_obs)
    return u * RANGES, y.reshape(-1, 1)


def run_eval(
    config: Metric12dConfig | None = None, models: tuple[str, ...] = MODELS
) -> list[CheckpointSummary]:
    print(f"ranges={','.join(f'{s:.4g}' for s in RANGES)}", flush=True)
    return run_stream_eval(config, models=models, data=make_data)
