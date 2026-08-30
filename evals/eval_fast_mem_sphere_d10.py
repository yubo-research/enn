from __future__ import annotations

from enn.turbo.config.enn_index_driver import ENNIndexDriver
from evals.flat_sphere import FlatSphereConfig, run_flat_sphere_eval


def evaluate() -> None:
    """FAST_MEM ENN on 10-D sphere: n=10*num_dim train, 100 test, 30 seeds; mean±se loglik/rmse."""
    run_flat_sphere_eval(FlatSphereConfig(index_driver=ENNIndexDriver.FAST_MEM))
