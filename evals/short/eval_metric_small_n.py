from __future__ import annotations

from evals.metric_small_n import run_eval


def evaluate() -> None:
    """Four 12-d targets x 5 seeds, n <= 300: BPANN_DISK NONE vs SCALE_X vs metric_learning ON vs AUTO
    loglik, nrmse, held-out gain."""
    run_eval()
