from __future__ import annotations

from enn.turbo.config.enn_index_driver import ENNIndexDriver
from evals.flat_sphere import FlatSphereConfig, run_flat_sphere_eval

NUM_DIM = 1000


def evaluate() -> None:
    """FAST_MEM ENN on 1000-D sphere: n=10*num_dim train, 100 test, 30 seeds; mean±se loglik/nrmse/rcorr."""
    run_flat_sphere_eval(
        FlatSphereConfig(
            num_dim=NUM_DIM,
            num_obs=10 * NUM_DIM,
            index_driver=ENNIndexDriver.FAST_MEM,
        )
    )
