import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("RAYON_NUM_THREADS", "1")
sys.path[:0] = ["/home/dsweet/projects/enn/src", "/home/dsweet/projects/enn"]
from enn import _rust
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from ops.stress import DRAW_FLAGS

D = 1000
wd, d, N, nq, out = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    sys.argv[5],
)
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.01
)
A = np.random.default_rng(7).standard_normal((d, D)) / 2


def gen(n, r):
    z = r.standard_normal((n, d))
    x = z @ A + 0.1 * r.standard_normal((n, D))
    f = np.sin(2 * z[:, 0]) + np.sin(2 * z[:, 1])
    return x, f, (f + 0.1 * r.standard_normal(n)).reshape(-1, 1)


if os.path.exists(os.path.join(wd, "CURRENT")) or (
    os.path.isdir(wd) and os.listdir(wd)
):
    m = EpistemicNearestNeighbors.open(wd)
else:
    os.makedirs(wd, exist_ok=True)
    m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
    m.add(np.empty((0, D)), np.empty((0, 1)))
    r = np.random.default_rng(11)
    t = time.time()
    for _ in range(N // 1000):
        x, _, y = gen(1000, r)
        m.add(x, y)
        m.schedule_background_flush()
    m.ensure_index_sync()
    print("built", time.time() - t, flush=True)
    time.sleep(3)


def drop():
    for root, _, fs in os.walk(wd):
        for f in fs:
            fd = os.open(os.path.join(root, f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


x, f, y = gen(nq, np.random.default_rng(1234))
res = {}
for c in (30.0, 10.0, 3.0):
    _rust.set_leaf_coefficient(c)
    drop()
    ts = []
    mu = []
    se = []
    _rust.take_walk_totals()
    for i in range(nq):
        t = time.perf_counter()
        o = m.posterior(x[i : i + 1], params=P, flags=DRAW_FLAGS)
        ts.append(time.perf_counter() - t)
        mu.append(float(np.asarray(o.mu).ravel()[0]))
        se.append(float(np.asarray(o.se).ravel()[0]))
    tot = _rust.take_walk_totals()
    mu = np.array(mu)
    se = np.array(se)
    yy = y.ravel()
    ll = -0.5 * np.log(2 * np.pi * se**2) - 0.5 * ((yy - mu) / se) ** 2
    sq = (yy - mu) ** 2
    res[c] = (ll, sq)
    ll0, sq0 = res[30.0]
    dl = ll - ll0
    ds = sq - sq0
    print(
        json.dumps(
            {
                "d": d,
                "N": N,
                "c": c,
                "loglik": float(ll.mean()),
                "loglik_se": float(ll.std(ddof=1) / np.sqrt(nq)),
                "d_loglik": float(dl.mean()),
                "d_loglik_se": float(dl.std(ddof=1) / np.sqrt(nq)),
                "nrmse": float(np.sqrt(sq.mean()) / yy.std()),
                "d_mse": float(ds.mean()),
                "d_mse_se": float(ds.std(ddof=1) / np.sqrt(nq)),
                "med_ms": float(np.median(ts)) * 1e3,
                "mean_ms": float(np.mean(ts)) * 1e3,
                "leaves_q": tot[0] / nq,
                "budget_stops_q": tot[2] / nq,
            }
        ),
        flush=True,
    )
