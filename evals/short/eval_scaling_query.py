from __future__ import annotations

from evals.scaling_n import run_eval


def evaluate() -> None:
    """BPANN_DISK + AUTO + OLS, 12-d stream to n=1e5, 3 seeds: posterior time on 1000 test
    points per checkpoint and per point, and its log-log slope in n."""
    run_eval("query")
