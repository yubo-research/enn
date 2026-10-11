from __future__ import annotations

from evals.stress_eval import TS_NUM_OBS_SWEEP, run_draw_stress_eval


def evaluate() -> None:
    for num_obs in TS_NUM_OBS_SWEEP:
        run_draw_stress_eval(num_obs=num_obs)
