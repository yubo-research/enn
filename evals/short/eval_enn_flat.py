from __future__ import annotations

from evals.stress_eval import NUM_OBS, run_enn_stress_eval


def evaluate() -> None:
    run_enn_stress_eval(index_type="flat", num_obs=NUM_OBS)
