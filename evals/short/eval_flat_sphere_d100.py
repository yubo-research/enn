from __future__ import annotations

from evals.flat_sphere import FlatSphereConfig, run_flat_sphere_eval

NUM_DIM = 100


def evaluate() -> None:
    run_flat_sphere_eval(FlatSphereConfig(num_dim=NUM_DIM, num_obs=10 * NUM_DIM))
