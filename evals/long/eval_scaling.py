"""Disk-backed streaming cost and accuracy through 100,000 rows over three seeds."""

from enn.evals.scaling_n import run_eval


def evaluate() -> None:
    run_eval()
