import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_util

sys.path.insert(0, "/home/dsweet/projects/enn/src")
sys.path.insert(0, "/home/dsweet/projects/enn")
from enn import _rust
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fitter import ENNStatefulFitter
from evals.harness.metrics import gaussian_loglik, rmse
from ops.stress import DRAW_FLAGS

kind, seed, wd, N = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
D, NT = 1000, 300
A = np.random.default_rng(7).standard_normal((4, D)) / 2.0
w = np.random.default_rng(8).standard_normal(D)


def gen(n, r):
    return probe_util.gen_sweep(kind, n, r, D, A, w)


rng = np.random.default_rng(seed)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
n = 0
while n < N:
    b = min(1000, N - n)
    x, y = gen(b, rng)
    m.add(x, y)
    m.schedule_background_flush()
    n += b
m.ensure_index_sync()
xt, yt = gen(NT, np.random.default_rng(seed + 999))
fitter = ENNStatefulFitter(k=10, seed=seed + 1)
_rust.set_leaf_coefficient(30.0)
params = fitter.ask(m, num_fit_candidates=16, num_fit_samples=32)
COEF = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0
for e in (0.0, 1.0, 3.0, 10.0, 30.0):
    _rust.set_leaf_coefficient(COEF)
    _rust.set_stop_epsilon(e)
    _rust.take_walk_totals()
    t0 = time.perf_counter()
    p = m.posterior(xt, params=params, flags=DRAW_FLAGS)
    dt = (time.perf_counter() - t0) / NT
    tot = _rust.take_walk_totals()
    mu = np.asarray(p.mu).ravel()
    se = np.asarray(p.se).ravel()
    y = yt.ravel()
    print(
        json.dumps(
            {
                "kind": kind,
                "N": N,
                "coef": COEF,
                "eps": e,
                "rmse": rmse(y, mu),
                "loglik": gaussian_loglik(y, mu, se),
                "sqerr": ((mu - y) ** 2).tolist(),
                "s_per_query": dt,
                "walk_totals": list(tot),
            }
        ),
        flush=True,
    )
