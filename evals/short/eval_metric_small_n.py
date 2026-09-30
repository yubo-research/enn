from __future__ import annotations

from evals.metric_small_n import run_eval


def evaluate() -> None:
    """Four 12-d targets x 5 seeds, n <= 300: BPANN_DISK NONE vs SCALE_X vs metric_learning AUTO
    loglik, nrmse, AUTO's leave-one-out gain."""
    run_eval()
