"""Which weights does the LOOCV metric fit learn on the different-ranges problem?

Fits the diagonal metric (reports/iaml/iaml_core.fit_exact, k=10, cold start at log a = 0, as
in the eval) on raw x and on u = x / s. The LOOCV objective is unchanged by x -> u, a -> a s^2,
so the fit on u is the raw-x fit started at log a = -2 log s. If the fit recovers the ranges,
log(a_i s_i^2) is the same for every input.
Usage: PYTHONPATH=src:. python reports/metric_learning/weights_check.py > weights_check.out
"""

import numpy as np

from evals.metric_12d import load_iaml_core
from evals.metric_ranges import RANGES, make_data

K = 10


def main():
    core = load_iaml_core()
    log_s2 = 2 * np.log(RANGES)
    for seed in range(3):
        for n in (300, 1000, 3000):
            x, y = make_data(n, np.random.default_rng(seed))
            for label, xx, shift in (("raw", x, log_s2), ("unit", x / RANGES, 0.0)):
                metric = core.Metric(xx.shape[1])
                core.fit_exact(metric, xx, y[:, 0], K)
                nbr = core.knn(xx, xx, K, metric.a, self_offset=0)
                loo = core.mean_ll(xx, y[:, 0], xx, y[:, 0], nbr, metric.a, metric.c)
                rel = metric.theta[:-1] + shift
                raw_log_a = rel - log_s2
                print(
                    f"seed={seed} n={n} start={label} loo={loo:+.3f} "
                    f"log(a*s^2)-max={np.round(rel - rel.max(), 1).tolist()} "
                    f"raw_log_a=[{raw_log_a.min():+.1f},{raw_log_a.max():+.1f}]"
                )


if __name__ == "__main__":
    main()
