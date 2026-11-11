import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors

kind, seed, wd, maxn = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
D, K, NQ = 1000, 10, 100
rng = np.random.default_rng(seed)
A = np.random.default_rng(7).standard_normal((16, D)) / 4.0


def gen(n, r):
    if kind == "gauss":
        return r.standard_normal((n, D))
    return r.standard_normal((n, 16)) @ A + 0.1 * r.standard_normal((n, D))


G = [g for g in (1000, 3000, 10000, 30000, 100000) if g <= maxn]
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
m.add(np.empty((0, D)), np.empty((0, 1)))
X = np.empty((0, D), dtype=np.float32)
n = 0
qr = np.random.default_rng(seed + 999)
for t in G:
    new = []
    while n < t:
        b = min(1000, t - n)
        x = gen(b, rng)
        new.append(x.astype(np.float32))
        m.add(x, rng.standard_normal((b, 1)))
        m.schedule_background_flush()
        n += b
    X = np.vstack([X] + new)
    m.ensure_index_sync()
    Q = gen(NQ, qr)
    t0 = time.perf_counter()
    got = [m.neighbors(q[None, :], K) for q in Q]
    dt = (time.perf_counter() - t0) / NQ
    rec = []
    Xn = (X.astype(np.float64) ** 2).sum(1)
    for q, g in zip(Q, got):
        d = Xn - 2 * X @ q + (q @ q)
        ex = set(np.argpartition(d, K)[:K].tolist())
        rec.append(len(ex & set(np.asarray(g).ravel().tolist())) / K)
    print(
        json.dumps(
            {
                "kind": kind,
                "N": n,
                "recall": float(np.mean(rec)),
                "se": float(np.std(rec, ddof=1) / np.sqrt(NQ)),
                "s_per_query": dt,
            }
        ),
        flush=True,
    )
