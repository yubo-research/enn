from __future__ import annotations

from evals.scaling_n import run_eval


def evaluate() -> None:
    """BPANN_DISK + AUTO + OLS, 12-d stream to n=1e5, 3 seeds: memory (resident, heap,
    memory-mapped, peak, disk), add, fit and query time, and test accuracy per checkpoint; then
    OLS of each cost metric on ln N, N and N^2 with t tests and backward elimination."""
    run_eval()
