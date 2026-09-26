from __future__ import annotations

from evals.flat_sphere import FlatSphereConfig, run_flat_sphere_eval

NUM_DIM = 100


def evaluate() -> None:
    """FLAT ENN on 100-D sphere: n=10*num_dim train, 100 test, 30 seeds; mean±se loglik/rmse."""
    run_flat_sphere_eval(FlatSphereConfig(num_dim=NUM_DIM, num_obs=10 * NUM_DIM))
