from __future__ import annotations

from evals.metric_12d import N_GRID, Metric12dConfig, run_eval

N_MAX = 100_000
CONFIG = Metric12dConfig(n_grid=tuple(n for n in N_GRID if n <= N_MAX), num_rows=max(N_GRID))


def evaluate() -> None:
    """long/metric_12d truncated to 1e5 rows: 12-d two-of-twelve stream, 10 seeds: mean±se
    loglik, nrmse, add_s, query_s per model and checkpoint."""
    run_eval(CONFIG)
