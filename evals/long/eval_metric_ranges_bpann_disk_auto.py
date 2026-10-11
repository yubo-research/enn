from __future__ import annotations

from evals.metric_ranges import run_eval

MODEL = "bpann_disk_auto"


def evaluate() -> None:
    run_eval(models=(MODEL,))
