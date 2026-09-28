"""Streaming eval on a 12-d problem whose inputs have very different ranges.

u ~ U[0,1]^12 and x_i = s_i u_i with ranges s_i = 10^linspace(-2, 2, 12), so the widest input
spans 10^4 times the narrowest. The target y = sum_i (u_i - 0.5) + 0.1 eps depends on every
input equally in its own units, so the ideal diagonal metric is known: weights a_i ∝ 1 / s_i^2,
which is what ``scale_x=ON`` applies up to a common factor. Unweighted distances are dominated by
the widest inputs. Models, streaming protocol, seeds and output lines are those of
``evals/metric_12d.py``; only the data differ.
"""

from __future__ import annotations

import numpy as np

from evals.metric_12d import NUM_DIM, Metric12dConfig, run_eval as run_stream_eval
from evals.metric_12d_results import CheckpointSummary

NOISE_STD = 0.1
RANGES = 10.0 ** np.linspace(-2.0, 2.0, NUM_DIM)


def make_data(num_obs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.random((num_obs, NUM_DIM))
    y = (u - 0.5).sum(axis=1) + NOISE_STD * rng.standard_normal(num_obs)
    return u * RANGES, y.reshape(-1, 1)


def run_eval(config: Metric12dConfig | None = None) -> list[CheckpointSummary]:
    print(f"ranges={','.join(f'{s:.4g}' for s in RANGES)}", flush=True)
    return run_stream_eval(config, data=make_data)
