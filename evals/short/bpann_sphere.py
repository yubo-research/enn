from __future__ import annotations

import tempfile

from enn.turbo.config.enn_index_driver import ENNIndexDriver
from evals.flat_sphere import FlatSphereConfig, run_flat_sphere_eval


def evaluate_bpann_disk_sphere(num_dim: int, work_dir_prefix: str) -> None:
    with tempfile.TemporaryDirectory(prefix=work_dir_prefix) as work_dir:
        run_flat_sphere_eval(
            FlatSphereConfig(
                num_dim=num_dim,
                num_obs=10 * num_dim,
                index_driver=ENNIndexDriver.BPANN_DISK,
                work_dir=work_dir,
            )
        )
