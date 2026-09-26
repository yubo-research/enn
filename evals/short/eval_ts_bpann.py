from __future__ import annotations

import tempfile

from evals.stress_eval import TS_NUM_OBS_SWEEP, run_draw_stress_eval

WORK_DIR_PREFIX = "enn_ts_bpann_eval_"


def evaluate() -> None:
    """Sweep draw stress over NUM_OBS with BPANN_DISK; prefix posterior rows with ``EVAL:``."""
    for num_obs in TS_NUM_OBS_SWEEP:
        with tempfile.TemporaryDirectory(prefix=WORK_DIR_PREFIX) as work_dir:
            run_draw_stress_eval(
                num_obs=num_obs,
                index_type="bpann_disk",
                work_dir=work_dir,
            )
