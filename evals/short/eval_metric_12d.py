from __future__ import annotations

from evals.metric_12d import run_eval


def evaluate() -> None:
    """12-d two-of-twelve stream to 1e6 rows: loglik, nrmse, add_s, query_s per model and checkpoint."""
    run_eval()
