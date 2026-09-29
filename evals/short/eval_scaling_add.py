from __future__ import annotations

from evals.scaling_n import run_eval


def evaluate() -> None:
    """BPANN_DISK + AUTO + OLS, 12-d stream to n=1e5, 3 seeds: add time per checkpoint and per
    row, fit time, and the log-log slope in n of add time per row."""
    run_eval("add")
