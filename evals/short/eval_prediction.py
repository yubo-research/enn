"""Held-out NRMSE, log-likelihood, build/fit time, and query time over three seeds."""

from enn.evals.prediction import SHORT_CONFIG, run_prediction


def evaluate() -> None:
    run_prediction(SHORT_CONFIG)
