from __future__ import annotations

from evals.metric_12d import run_eval

MODEL = "flat_scale_x"


def evaluate() -> None:
    """12-d two-of-twelve stream to 1e6 rows, 10 seeds, model flat_scale_x only:
    mean±se loglik, nrmse, add_s, query_s per checkpoint."""
    run_eval(models=(MODEL,))
