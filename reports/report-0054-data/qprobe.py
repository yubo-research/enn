import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams

seed, wd, maxn = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
D, K, NQ = 1000, 10, 60
P = ENNParams(
    k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
grid = [
    g
    for g in (1000, 3000, 10000, 30000, 100000, 200000, 300000, 500000, 700000, 1000000)
    if g <= maxn
]
rng, qr = np.random.default_rng(seed), np.random.default_rng(seed + 999)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
m.add(np.empty((0, D)), np.empty((0, 1)))
n = 0


def rss():
    d = {}
    for l in Path("/proc/self/status").read_text().splitlines():
        k, _, v = l.partition(":")
        if k in ("RssAnon", "RssFile"):
            d[k] = int(v.split()[0]) / 1024
    return d


for t in grid:
    while n < t:
        b = min(1000, t - n)
        m.add(rng.standard_normal((b, D)), rng.standard_normal((b, 1)))
        m.schedule_background_flush()
        n += b
    m.ensure_index_sync()
    m.posterior(qr.standard_normal((1, D)), params=P)
    ts = []
    for _ in range(NQ):
        q = qr.standard_normal((1, D))
        t0 = time.perf_counter()
        m.posterior(q, params=P)
        ts.append(time.perf_counter() - t0)
    print(json.dumps({"n": t, "seed": seed, "q1": ts, "rss": rss()}), flush=True)
