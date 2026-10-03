from __future__ import annotations

from evals.metric_12d import N_GRID, Metric12dConfig
from evals.metric_ranges import run_eval

N_MAX = 100_000
CONFIG = Metric12dConfig(n_grid=tuple(n for n in N_GRID if n <= N_MAX), num_rows=max(N_GRID))


def evaluate() -> None:
    """long/metric_ranges_* truncated to 1e5 rows, all models in one run: 12-d stream, input
    ranges 1e-2..1e2, all inputs relevant, 10 seeds: mean±se loglik, nrmse, add_s, query_s per
    model and checkpoint."""
    run_eval(CONFIG)
