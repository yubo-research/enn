from enn.evals.prediction import SHORT_CONFIG, run_prediction


def evaluate() -> None:
    run_prediction(SHORT_CONFIG)
