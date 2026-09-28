"""Offline check of streaming metric weights (no index): held-out ENN log-likelihood by exact kNN.

For each problem and seed, rows stream in batches of 500 to ``N``. Weights compared at each
checkpoint: equal (NONE), 1/Var (SCALE_X), dependence weights from the reservoir (correlation,
Sobol), the SPSA learner, and the L-BFGS-B LOOCV fit (iaml_core.fit_exact) on the reservoir.
Each weight vector is scored by the mean Gaussian log-likelihood of 2,000 test rows predicted
from their k nearest of the first ``n`` rows, with the noise and epistemic scale refit by a
1-d grid over a common factor on the weights and c (a crude stand-in for enn_fit).
Usage: PYTHONPATH=src:. python reports/metric_learning/stream_weights_check.py 12d|ranges N 0,1,2 [--no-lbfgs]
"""

import sys
import time

import numpy as np

from enn.enn.metric_stream import (
    PerturbationMetricLearner,
    Reservoir,
    dependence_weights,
    validated_dependence_weights,
)
from evals.metric_12d import load_iaml_core, make_data as data_12d
from evals.metric_ranges import make_data as data_ranges

K = 10
INIT_ROWS = 100
NUM_TEST = 2000
BATCH = 500
CHECKPOINTS = (100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000)


def score(core, a, x, y, xt, yt):
    """Best mean test loglik over a grid of common factors on a and of noise c (y is ~unit variance)."""
    best = -np.inf
    for f in 10.0 ** np.arange(-3, 3.01, 0.5):
        nbr = core.knn(xt, x, K, a * f)
        for c in (0.003, 0.01, 0.03, 0.1, 0.3):
            best = max(best, core.mean_ll(xt, yt, x, y, nbr, a * f, c))
    return best


def candidates(core, res, lrn, lrn_s, x_seen, lbfgs):
    rx, ry = res.x, res.y
    cands = {
        "none": np.ones(rx.shape[1]),
        "scale_x": 1.0 / x_seen.var(0),
        "corr": dependence_weights(rx, ry, "correlation"),
        "sobol": dependence_weights(rx, ry, "sobol"),
        "sobol_v": validated_dependence_weights(rx, ry, K, "sobol"),
        "corr_v": validated_dependence_weights(rx, ry, K, "correlation"),
        "spsa": lrn.weights,
        "spsa_sobol": lrn_s.weights,
    }
    if lbfgs:
        m = core.Metric(rx.shape[1])
        core.fit_exact(m, rx, ry, K)
        cands["lbfgs"] = m.a
    return cands


def stream(res, lrn, lrn_s, x, y, lo, hi):
    """Feed rows lo:hi in batches; return seconds spent in the plain SPSA learner."""
    t_spsa = 0.0
    for s in range(lo, hi, BATCH):
        e = min(hi, s + BATCH)
        res.add(x[s:e], y[s:e])
        t0 = time.perf_counter()
        lrn.update(x[s:e], y[s:e])
        t_spsa += time.perf_counter() - t0
        lrn_s.update(x[s:e], y[s:e])
    return t_spsa


def run(problem, n_max, seed, lbfgs):
    core = load_iaml_core()
    rng = np.random.default_rng(seed)
    make = data_12d if problem == "12d" else data_ranges
    x, y2 = make(n_max, rng)
    xt, yt2 = make(NUM_TEST, rng)
    y, yt = y2[:, 0], yt2[:, 0]
    res = Reservoir(1000, x.shape[1], np.random.default_rng(seed + 1))
    lrn = PerturbationMetricLearner(x.shape[1], np.random.default_rng(seed + 2))
    lrn_s = PerturbationMetricLearner(x.shape[1], np.random.default_rng(seed + 2), init_rows=INIT_ROWS)
    t_spsa, lo = 0.0, 0
    for hi in [n for n in CHECKPOINTS if n <= n_max]:
        t_spsa += stream(res, lrn, lrn_s, x, y, lo, hi)
        lo = hi
        cands = candidates(core, res, lrn, lrn_s, x[:hi], lbfgs)
        xs, ys = x[: min(hi, 20000)], y[: min(hi, 20000)]
        out = " ".join(f"{k}={score(core, a, xs, ys, xt, yt):+.3f}" for k, a in cands.items())
        print(f"{problem} seed={seed} n={hi} {out} spsa_s={t_spsa:.1f}", flush=True)
        for name, lr in (("spsa", lrn), ("spsa_sobol", lrn_s)):
            print(f"  {name} log(a*var)={np.round(lr.theta[:-1], 2).tolist()}", flush=True)


if __name__ == "__main__":
    problem = sys.argv[1]
    n_max = int(sys.argv[2])
    seeds = [int(s) for s in sys.argv[3].split(",")]
    for seed in seeds:
        run(problem, n_max, seed, "--no-lbfgs" not in sys.argv)
