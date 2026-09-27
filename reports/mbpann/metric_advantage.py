"""A test problem on which BPANN_DISK + metric learning predicts out of sample better than BPANN_DISK
because of metric learning.

Problem ("two_of_twelve"): x ~ U[0,1]^12, y = sin(6 pi x0) + sin(6 pi x1) + 0.1 eps.
Ten of the twelve inputs are irrelevant, so under the identity metric most of a
query's neighbors are close in the irrelevant inputs and far in x0, x1.

Stream: 20,000 rows in batches of 500, `ensure_index_sync` after each batch. At this
batch size BPANN searches every leaf (recall 1.0), so index approximation cannot
explain a difference. Every 2,000 rows, BPANN_DISK + metric learning refits diagonal metric weights by
safeguarded LOOCV (reports/iaml/iaml_core.fit_exact) on a 1,000-row subsample of the rows
seen so far and applies them with MBPANNMetric.set_weights.

Models (all use the ENN posterior from batch_posterior, k = 10):
- bpann_disk: BPANN_DISK, identity metric (it cannot change its metric in place).
- mbpann_identity: BPANN_DISK + metric learning, never given weights (control: the mode alone).
- mbpann_learned: BPANN_DISK + metric learning with the learned weights.

ENN variance scales are chosen per model on 1,000 validation rows (grid), then the mean
Gaussian log-likelihood and RMSE are reported on 2,000 separate test rows.
Recall@10 is against exact neighbors under each model's own metric.

Usage: PYTHONPATH=src:reports/iaml python reports/mbpann/metric_advantage.py OUT.json [seeds]
"""

import json
import sys
import tempfile

import numpy as np
from iaml_core import Metric, fit_exact, knn

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags
from enn.enn.mbpann import MBPANNMetric
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_x_scaling import ENNXScaling

D, K = 12, 10
N_TRAIN, N_VAL, N_TEST = 20000, 1000, 2000
BATCH, REFIT_EVERY, FIT_SUBSAMPLE = 500, 2000, 1000
EPI_GRID = np.logspace(-3, 3, 13)
ALE_GRID = np.logspace(-4, 0, 9)
GRID = [(e, a) for e in EPI_GRID for a in ALE_GRID]


def f(x):
    return np.sin(6 * np.pi * x[:, 0]) + np.sin(6 * np.pi * x[:, 1])


def make_data(seed):
    rng = np.random.default_rng(seed)
    x = rng.random((N_TRAIN + N_VAL + N_TEST, D))
    y = f(x) + 0.1 * rng.standard_normal(len(x))
    y = (y - y[:N_TRAIN].mean()) / y[:N_TRAIN].std()
    s = np.cumsum([0, N_TRAIN, N_VAL, N_TEST])
    return [(x[lo:hi], y[lo:hi]) for lo, hi in zip(s[:-1], s[1:])], rng


def new_model(x, y, x_scaling):
    return EpistemicNearestNeighbors(
        x, y.reshape(-1, 1), x_scaling=x_scaling, index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tempfile.mkdtemp(prefix="madv_")
    )


def gaussian_ll(mu, se, y):
    return -0.5 * np.log(2 * np.pi * se**2) - 0.5 * ((y - mu) / se) ** 2


def predict(model, xq):
    params = [ENNParams(k_num_neighbors=K, epistemic_variance_scale=e, aleatoric_variance_scale=a) for e, a in GRID]
    post = model.batch_posterior(xq, params, flags=PosteriorFlags())
    return post.mu[:, :, 0], post.se[:, :, 0], np.asarray(post.idx, dtype=np.int64)


def score(model, weights, train, val, test):
    mu_v, se_v, _ = predict(model, val[0])
    best = int(np.argmax(gaussian_ll(mu_v, se_v, val[1]).mean(1)))
    mu_t, se_t, idx_t = predict(model, test[0])
    ex = knn(test[0], train[0], K, weights)
    return dict(
        test_ll=float(gaussian_ll(mu_t[best], se_t[best], test[1]).mean()),
        test_rmse=float(np.sqrt(np.mean((mu_t[best] - test[1]) ** 2))),
        recall=float(np.mean([len(set(u) & set(v)) / K for u, v in zip(idx_t, ex)])),
        epi=float(GRID[best][0]),
        ale=float(GRID[best][1]),
    )


def run(seed):
    (train, val, test), rng = make_data(seed)
    x, y = train
    models = {
        "bpann_disk": new_model(x[:BATCH], y[:BATCH], ENNXScaling.NONE),
        "mbpann_identity": new_model(x[:BATCH], y[:BATCH], ENNXScaling.METRIC_LEARNING),
        "mbpann_learned": new_model(x[:BATCH], y[:BATCH], ENNXScaling.METRIC_LEARNING),
    }
    helper = MBPANNMetric(models["mbpann_learned"])
    metric = Metric(D)
    for m in models.values():
        m.ensure_index_sync()
    for lo in range(BATCH, N_TRAIN, BATCH):
        hi = lo + BATCH
        for m in models.values():
            m.add(x[lo:hi], y[lo:hi].reshape(-1, 1))
            m.ensure_index_sync()
        if hi % REFIT_EVERY == 0:
            sub = rng.choice(hi, size=min(hi, FIT_SUBSAMPLE), replace=False)
            fit_exact(metric, x[sub], y[sub], K)
            helper.set_weights(metric.a)
    ones = np.ones(D)
    out = {}
    for key, m in models.items():
        w = helper.weights if key == "mbpann_learned" else ones
        out[key] = score(m, w, train, val, test)
        print(seed, key, {k: round(v, 4) for k, v in out[key].items()}, flush=True)
    out["weights"] = helper.weights.tolist()
    out["rebuilds"], out["rescales"] = helper.num_rebuilds, helper.num_rescales
    print(seed, "weights", np.round(helper.weights, 3).tolist(), flush=True)
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    results = {str(s): run(s) for s in seeds}
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
