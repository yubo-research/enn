from __future__ import annotations

from evals.metric_12d import N_GRID, Metric12dConfig
from evals.metric_ranges import run_eval

N_MAX = 100_000
CONFIG = Metric12dConfig(n_grid=tuple(n for n in N_GRID if n <= N_MAX), num_rows=max(N_GRID))


def evaluate() -> None:
    run_eval(CONFIG)
