from __future__ import annotations

import tempfile

from enn.turbo.config.enn_index_driver import ENNIndexDriver
from evals.flat_sphere import FlatSphereConfig, run_flat_sphere_eval

NUM_DIM = 100
WORK_DIR_PREFIX = "enn_bpann_sphere_d100_"


def evaluate() -> None:
    """BPANN_DISK ENN on 100-D sphere: n=10*num_dim train, 100 test, 30 seeds; mean±se loglik/nrmse/rcorr."""
    with tempfile.TemporaryDirectory(prefix=WORK_DIR_PREFIX) as work_dir:
        run_flat_sphere_eval(
            FlatSphereConfig(
                num_dim=NUM_DIM,
                num_obs=10 * NUM_DIM,
                index_driver=ENNIndexDriver.BPANN_DISK,
                work_dir=work_dir,
            )
        )
