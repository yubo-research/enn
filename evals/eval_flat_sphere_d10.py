from __future__ import annotations

from evals.flat_sphere import run_flat_sphere_eval


def evaluate() -> None:
    """FLAT ENN on 10-D sphere: n=10*num_dim train, 100 test, 30 seeds; mean±se loglik/rmse."""
    run_flat_sphere_eval()
