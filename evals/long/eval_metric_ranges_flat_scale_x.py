from __future__ import annotations

from evals.metric_ranges import run_eval

MODEL = "flat_scale_x"


def evaluate() -> None:
    """12-d stream to 1e6 rows, input ranges 1e-2..1e2, all inputs relevant,
    10 seeds, model flat_scale_x only: mean±se loglik, nrmse, add_s, query_s per checkpoint."""
    run_eval(models=(MODEL,))
