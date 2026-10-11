from __future__ import annotations

from evals.short.bpann_sphere import evaluate_bpann_disk_sphere

NUM_DIM = 1000
WORK_DIR_PREFIX = "enn_bpann_sphere_d1000_"


def evaluate() -> None:
    evaluate_bpann_disk_sphere(NUM_DIM, WORK_DIR_PREFIX)
