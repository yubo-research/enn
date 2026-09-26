from __future__ import annotations

import tempfile

from evals.stress_eval import NUM_OBS, run_enn_stress_eval

WORK_DIR_PREFIX = "enn_bpann_disk_eval_"


def evaluate() -> None:
    """Run bpann_disk ENN stress and prefix metric rows with ``EVAL:``."""
    with tempfile.TemporaryDirectory(prefix=WORK_DIR_PREFIX) as work_dir:
        run_enn_stress_eval(
            index_type="bpann_disk",
            num_obs=NUM_OBS,
            work_dir=work_dir,
        )
