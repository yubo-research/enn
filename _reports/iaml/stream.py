"""Streaming comparison of metric-learning strategies for ENN with a BPANN index.

Usage: python stream.py OUT.json [seeds]
"""

import json
import sys
import time

import numpy as np

from iaml_core import Adam, BpannPool, Metric, fit_exact, fit_on_pools, mean_ll, rerank

K = 10
BATCH = 100
N_TRAIN = 3100
N_TEST = 2000
CHECKPOINTS = (300, 700, 1500, 3100)
EPOCHS = {100, 200, 400, 800, 1600}


def f_sparse(x):
    return np.sin(6 * x[:, 0]) + np.cos(4 * x[:, 1]) + 0.5 * x[:, 2], 0.1


def f_friedman(x):
    return (
        10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2 + 10 * x[:, 3] + 5 * x[:, 4],
        1.0,
    )


def f_aniso(x):
    s = np.array([4, 2, 1, 0.5, 0.25, 0.125])
    return np.sin(2 * np.pi * x * s).sum(1), 0.1


def f_iso(x):
    return np.sin(2 * np.pi * x).sum(1), 0.1


PROBLEMS = {"sparse": (f_sparse, 10), "friedman": (f_friedman, 10), "aniso": (f_aniso, 6), "iso": (f_iso, 5)}


def make_data(name, seed):
    f, d = PROBLEMS[name]
    rng = np.random.default_rng(seed)
    x = rng.random((N_TRAIN + N_TEST, d))
    mu, sd = f(x)
    y = mu + sd * rng.standard_normal(len(x))
    y = (y - y[:N_TRAIN].mean()) / y[:N_TRAIN].std()
    return x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:]


class Exact:
    """Exact neighbors in the current metric: equivalent to rebuilding the index after each refit."""

    def __init__(self, d, iso, refit_at=None):
        self.m = Metric(d, isotropic=iso)
        self.refit_at = refit_at
        self.rows_indexed = 0
        self.index = None

    def update(self, x, y, n):
        if self.m.iso and self.index is not None:
            fit_exact(self.m, x[:n], y[:n], K)
            self.index.add(x[n - BATCH : n], y[n - BATCH : n])
            self.rows_indexed += BATCH
        elif self.refit_at is None or n in self.refit_at:
            fit_exact(self.m, x[:n], y[:n], K)
            self.index = BpannPool(x[:n], y[:n], scale=self.m.a)
            self.rows_indexed += n
        else:
            self.index.add(x[n - BATCH : n], y[n - BATCH : n])
            self.rows_indexed += BATCH

    def loo_nbr(self, x, y, n):
        return self.index.query(x[:n], K, exclude_self=True)

    def test_nbr(self, x, n, xt):
        return self.index.query(xt, K, exclude_self=False)


class Pooled:
    """Raw-coordinate BPANN never rebuilt; candidates re-ranked by the learned metric."""

    def __init__(self, d, mult, online=False):
        self.m = Metric(d)
        self.mult = mult
        self.online = online
        self.index = None
        self.rows_indexed = 0
        self.opt = Adam(d + 1)

    def update(self, x, y, n):
        lo = n - BATCH
        if self.index is None:
            self.index = BpannPool(x[:n], y[:n])
            self.rows_indexed += n
            fit_on_pools(self.m, x[:n], y[:n], self.index.query(x[:n], K * self.mult, True), K)
            return
        if self.online:
            pool = self.index.query(x[lo:n], K * self.mult, exclude_self=False)
            for i in range(lo, n):
                p = pool[i - lo : i - lo + 1]
                nbr = rerank(x[i : i + 1], x, p, K, self.m.a)
                self.m.sgd_step(x[i : i + 1], y[i : i + 1], x, y, nbr, self.opt)
        self.index.add(x[lo:n], y[lo:n])
        self.rows_indexed += BATCH
        if not self.online:
            fit_on_pools(self.m, x[:n], y[:n], self.index.query(x[:n], K * self.mult, True), K)

    def loo_nbr(self, x, y, n):
        return rerank(x[:n], x, self.index.query(x[:n], K * self.mult, True), K, self.m.a)

    def test_nbr(self, x, n, xt):
        return rerank(xt, x, self.index.query(xt, K * self.mult, False), K, self.m.a)


def methods(d):
    return {
        "iso": Exact(d, iso=True),
        "full": Exact(d, iso=False),
        "A_pool1": Pooled(d, 1),
        "A_pool3": Pooled(d, 3),
        "A_pool10": Pooled(d, 10),
        "B_doubling": Exact(d, iso=False, refit_at=EPOCHS),
        "C_online10": Pooled(d, 10, online=True),
    }


def run(name, seed):
    x, y, xt, yt = make_data(name, seed)
    ms = methods(x.shape[1])
    rows = []
    for key, meth in ms.items():
        t0 = time.time()
        for n in range(BATCH, N_TRAIN + 1, BATCH):
            meth.update(x, y, n)
            if n in CHECKPOINTS:
                loo = mean_ll(x[:n], y[:n], x, y, meth.loo_nbr(x, y, n), meth.m.a, meth.m.c)
                tst = mean_ll(xt, yt, x, y, meth.test_nbr(x, n, xt), meth.m.a, meth.m.c)
                rows.append(
                    dict(
                        problem=name,
                        seed=seed,
                        method=key,
                        n=n,
                        loo=loo,
                        test=tst,
                        rows_indexed=meth.rows_indexed,
                        seconds=time.time() - t0,
                        a=(meth.m.a / meth.m.a.max()).tolist(),
                        c=meth.m.c,
                    )
                )
        print(name, seed, key, f"loo={rows[-1]['loo']:.3f} test={rows[-1]['test']:.3f}", flush=True)
    return rows


if __name__ == "__main__":
    out = sys.argv[1]
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    allrows = []
    for name in PROBLEMS:
        for s in seeds:
            allrows += run(name, s)
            with open(out, "w") as fh:
                json.dump(allrows, fh)
