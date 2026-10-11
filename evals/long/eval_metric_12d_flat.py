from __future__ import annotations

from evals.metric_12d import run_eval

MODEL = "flat"


def evaluate() -> None:
    run_eval(models=(MODEL,))
