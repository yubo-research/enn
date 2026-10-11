from __future__ import annotations

from evals.stress_eval import run_turbo_enn_stress_eval


def evaluate() -> None:
    run_turbo_enn_stress_eval(index_type="flat", num_obs=1000, num_ask=100)
