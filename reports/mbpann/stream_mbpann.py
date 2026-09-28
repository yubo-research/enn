"""Compare index maintenance strategies when the ENN metric changes during streaming.

Every method replays the same metric sequence (fit once, up front), so differences
come only from how the BPANN index follows the metric.

Usage: PYTHONPATH=src:reports/iaml python reports/mbpann/stream_mbpann.py OUT.json [seeds] [problems]
"""

import json
import os
import sys
import tempfile
import time

import numpy as np
from iaml_core import Metric, fit_exact, knn, mean_ll
from stream import f_friedman, f_sparse

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags
from enn.enn.mbpann import MBPANNMetric
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_x_scaling import ENNMetricLearning

K = 10
BATCH = int(os.environ.get("MBPANN_BATCH", "2000"))
N_TRAIN = 40000
N_TEST = 2000
N_LOO = 2000
FIT_SUBSAMPLE = 2000
CHECKPOINTS = (10000, 20000, 30000, 40000)


PROBLEMS = {"sparse": (f_sparse, 10), "friedman": (f_friedman, 10)}


def make_data(name, seed):
    f, d = PROBLEMS[name]
    rng = np.random.default_rng(seed)
    x = rng.random((N_TRAIN + N_TEST, d))
    mu, sd = f(x)
    noisy = mu + sd * rng.standard_normal(len(x))
    ytr = noisy[:N_TRAIN]
    y = (noisy - ytr.mean()) / ytr.std()
    return x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:], rng


def metric_sequence(x, y, rng):
    """Metric (a, c) after each batch, fit by safeguarded LOOCV on a subsample."""
    m = Metric(x.shape[1])
    seq = {}
    for n in range(BATCH, N_TRAIN + 1, BATCH):
        sub = rng.choice(n, size=min(n, FIT_SUBSAMPLE), replace=False)
        fit_exact(m, x[sub], y[sub], K)
        seq[n] = (m.a.copy(), m.c)
    return seq


def query_idx(model, xq, exclude_self):
    p = ENNParams(k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1)
    flags = PosteriorFlags(exclude_nearest=exclude_self)
    return np.asarray(model.batch_posterior(xq, [p], flags=flags).idx, dtype=np.int64)


def new_model(x, y, metric_learning):
    return EpistemicNearestNeighbors(
        x, y.reshape(-1, 1), metric_learning=metric_learning, index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tempfile.mkdtemp(prefix="mbpann_")
    )


class BpannCopy:
    """BPANN_DISK: the only way to change metric is a new model on sqrt(a)*x."""

    def __init__(self, x, y):
        self.x, self.y, self.s, self.model = x, y, np.ones(x.shape[1]), None

    def add(self, lo, hi):
        if self.model is None:
            self.model = new_model(self.x[:hi] * self.s, self.y[:hi], ENNMetricLearning.OFF)
        else:
            self.model.add(self.x[lo:hi] * self.s, self.y[lo:hi].reshape(-1, 1))
        self.model.ensure_index_sync()

    def set_metric(self, a, n):
        self.s = np.sqrt(a)
        self.model = new_model(self.x[:n] * self.s, self.y[:n], ENNMetricLearning.OFF)
        self.model.ensure_index_sync()

    def query(self, xq, exclude_self):
        return query_idx(self.model, xq * self.s, exclude_self)


class Mbpann:
    def __init__(self, x, y, rebuild_drift):
        self.x, self.y, self.drift, self.model, self.metric = x, y, rebuild_drift, None, None

    def add(self, lo, hi):
        if self.model is None:
            self.model = new_model(self.x[:hi], self.y[:hi], ENNMetricLearning.ON)
            self.metric = MBPANNMetric(self.model, rebuild_drift=self.drift)
        else:
            self.model.add(self.x[lo:hi], self.y[lo:hi].reshape(-1, 1))
        self.model.ensure_index_sync()

    def set_metric(self, a, n):
        self.metric.set_weights(a)

    def query(self, xq, exclude_self):
        return query_idx(self.model, xq, exclude_self)


class Frozen(Mbpann):
    """Floor: identity metric forever (no metric learning)."""

    def __init__(self, x, y):
        super().__init__(x, y, np.inf)

    def set_metric(self, a, n):
        pass


class Oracle(BpannCopy):
    """BPANN_DISK streamed in the final metric from the start: never stale (valid at the last checkpoint)."""

    def __init__(self, x, y, a_final):
        super().__init__(x, y)
        self.s = np.sqrt(a_final)

    def set_metric(self, a, n):
        pass


def methods(x, y, a_final):
    return {
        "frozen_identity": Frozen(x, y),
        "oracle_final_metric": Oracle(x, y, a_final),
        "bpann_copy_rebuild": BpannCopy(x, y),
        "mbpann_rebuild": Mbpann(x, y, 0.0),
        "mbpann_auto": Mbpann(x, y, float(np.log(2.0))),
        "mbpann_rescale": Mbpann(x, y, np.inf),
    }


def evaluate(meth, metric, data, n, loo_rows):
    a_eval, c = metric
    x, y, xt, yt = data
    t0 = time.perf_counter()
    loo_idx = meth.query(x[loo_rows], exclude_self=True)
    test_idx = meth.query(xt, exclude_self=False)
    q_sec = time.perf_counter() - t0
    ex_loo = knn(x[loo_rows], x[:n], K + 1, a_eval)
    ex_loo = np.array([r[r != i][:K] for r, i in zip(ex_loo, loo_rows)])
    ex_test = knn(xt, x[:n], K, a_eval)
    recall = np.mean([len(set(u) & set(v)) / K for u, v in zip(test_idx, ex_test)])
    return dict(
        loo=mean_ll(x[loo_rows], y[loo_rows], x, y, loo_idx, a_eval, c),
        test=mean_ll(xt, yt, x, y, test_idx, a_eval, c),
        loo_exact=mean_ll(x[loo_rows], y[loo_rows], x, y, ex_loo, a_eval, c),
        test_exact=mean_ll(xt, yt, x, y, ex_test, a_eval, c),
        recall=float(recall),
        query_sec=q_sec,
    )


def run_method(key, meth, seq, data, loo_sets):
    x = data[0]
    t_add = t_metric = 0.0
    rows = []
    for n in range(BATCH, N_TRAIN + 1, BATCH):
        t0 = time.perf_counter()
        meth.add(n - BATCH, n)
        t1 = time.perf_counter()
        meth.set_metric(seq[n][0], n)
        t2 = time.perf_counter()
        t_add, t_metric = t_add + t1 - t0, t_metric + t2 - t1
        if n in CHECKPOINTS:
            a_eval = np.ones(x.shape[1]) if key == "frozen_identity" else seq[n][0]
            row = evaluate(meth, (a_eval, seq[n][1]), data, n, loo_sets[n])
            row.update(method=key, n=n, add_sec=t_add, metric_sec=t_metric)
            if isinstance(meth, Mbpann) and meth.metric is not None:
                row.update(rebuilds=meth.metric.num_rebuilds, rescales=meth.metric.num_rescales)
            rows.append(row)
    return rows


def run(name, seed):
    x, y, xt, yt, rng = make_data(name, seed)
    seq = metric_sequence(x, y, rng)
    loo_sets = {n: np.sort(rng.choice(n, size=N_LOO, replace=False)) for n in CHECKPOINTS}
    out = []
    for key, meth in methods(x, y, seq[N_TRAIN][0]).items():
        rows = run_method(key, meth, seq, (x, y, xt, yt), loo_sets)
        for r in rows:
            r.update(problem=name, seed=seed)
        last = rows[-1]
        print(
            name, seed, key,
            f"loo={last['loo']:.3f}/{last['loo_exact']:.3f} test={last['test']:.3f}/{last['test_exact']:.3f}",
            f"recall={last['recall']:.3f} add={last['add_sec']:.2f}s metric={last['metric_sec']:.2f}s "
            f"query={last['query_sec']:.2f}s",
            flush=True,
        )
        out += rows
    return out


if __name__ == "__main__":
    out = sys.argv[1]
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    probs = sys.argv[3].split(",") if len(sys.argv) > 3 else list(PROBLEMS)
    allrows = []
    for name in probs:
        for s in seeds:
            allrows += run(name, s)
            with open(out, "w") as fh:
                json.dump(allrows, fh)
