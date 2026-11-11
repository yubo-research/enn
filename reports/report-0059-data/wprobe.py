import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_util

from enn._rust import take_walk_totals
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams

seed, wd, maxn = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
D, K, NQ = 1000, 10, 60
P = ENNParams(
    k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
G = (1000, 3000, 10000, 30000, 100000, 200000, 300000, 500000, 700000, 1000000)
grid = [g for g in G if g <= maxn]
rng, qr = np.random.default_rng(seed), np.random.default_rng(seed + 999)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
m.add(np.empty((0, D)), np.empty((0, 1)))


io = probe_util.proc_io_counters


n = 0
for t in grid:
    while n < t:
        b = min(1000, t - n)
        m.add(rng.standard_normal((b, D)), rng.standard_normal((b, 1)))
        m.schedule_background_flush()
        n += b
    m.ensure_index_sync()
    m.posterior(qr.standard_normal((1, D)), params=P)
    take_walk_totals()
    ts, rows = [], []
    for _ in range(NQ):
        q = qr.standard_normal((1, D))
        a = io()
        take_walk_totals()
        t0 = time.perf_counter()
        m.posterior(q, params=P)
        dt = time.perf_counter() - t0
        w = take_walk_totals()
        b_ = io()
        rows.append(
            {
                "dt": dt,
                "w": w,
                "minflt": b_[0] - a[0],
                "majflt": b_[1] - a[1],
                "rbytes": b_[2] - a[2],
            }
        )
    print(json.dumps({"n": t, "seed": seed, "q": rows}), flush=True)
