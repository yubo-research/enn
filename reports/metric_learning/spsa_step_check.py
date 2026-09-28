"""Does a smaller SPSA step stop the learner from spoiling a good (Sobol) start?

Runs bpann_disk_sobol and bpann_disk_spsa_sobol at several SPSA steps on both streaming problems,
10 seeds, checkpoints to 1e4, and prints paired loglik differences spsa_sobol - sobol per step and n.
Values are paired within this run only (a shorter checkpoint grid draws a different test set than
the 1e6-row runs). Usage: PYTHONPATH=src:. python reports/metric_learning/spsa_step_check.py > spsa_step_check.out
"""

import tempfile
from dataclasses import replace

import numpy as np

from enn.enn.metric_stream import PerturbationMetricLearner
from evals import metric_12d, metric_ranges

STEPS = (0.05, 0.01, 0.002)
CFG = metric_12d.Metric12dConfig(n_grid=(10, 30, 100, 300, 1000, 3000, 10000))
PROBLEMS = {"12d": metric_12d.make_data, "ranges": metric_ranges.make_data}


def loglik(name, data, seed):
    with tempfile.TemporaryDirectory() as work_dir:
        results = metric_12d.run_model(name, work_dir, replace(CFG, seed=seed), data)
    return np.array([r.loglik for r in results])


def main():
    defaults = PerturbationMetricLearner.__init__.__kwdefaults__
    for tag, data in PROBLEMS.items():
        seeds = range(CFG.num_seeds)
        base = np.array([loglik("bpann_disk_sobol", data, s) for s in seeds])
        for step in STEPS:
            defaults["step"] = step
            diff = np.array([loglik("bpann_disk_spsa_sobol", data, s) for s in seeds]) - base
            for n, d in zip(CFG.n_grid, diff.T):
                se = d.std(ddof=1) / np.sqrt(len(d))
                print(f"RESULT {tag} step={step} n={n} spsa_sobol-sobol={d.mean():+.3f}±{se:.3f} wins={int((d > 0).sum())}", flush=True)
        defaults["step"] = STEPS[0]


if __name__ == "__main__":
    main()
