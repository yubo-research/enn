"""Prediction and wall-time curves to 100,000 rows over ten paired seeds."""

from enn.evals.prediction import LONG_CONFIG, run_prediction


def evaluate() -> None:
    run_prediction(LONG_CONFIG)
