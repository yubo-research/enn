from __future__ import annotations

from evals.short.bpann_sphere import evaluate_bpann_disk_sphere

NUM_DIM = 100
WORK_DIR_PREFIX = "enn_bpann_sphere_d100_"


def evaluate() -> None:
    evaluate_bpann_disk_sphere(NUM_DIM, WORK_DIR_PREFIX)
