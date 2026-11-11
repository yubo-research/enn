import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("RAYON_NUM_THREADS", "1")
sys.path.insert(0, "/home/dsweet/projects/enn/src")
from enn import _rust
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams

P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
D = 1000
A = np.random.default_rng(7).standard_normal((4, D)) / 2
seed = int(sys.argv[2])
maxn = int(sys.argv[3])


def gen(n, r):
    z = r.standard_normal((n, 4))
    x = z @ A + 0.1 * r.standard_normal((n, D))
    return x, (np.sin(z[:, 0]) + 0.1 * r.standard_normal(n)).reshape(-1, 1)


def anon():
    for l in Path("/proc/self/status").read_text().splitlines():
        if l.startswith("RssAnon"):
            return int(l.split()[1]) / 1024


r = np.random.default_rng(seed)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=sys.argv[1])
m.add(np.empty((0, D)), np.empty((0, 1)))
n = 0
qr = np.random.default_rng(5)
addt = 0.0
addn = 0
for tgt in [
    t for t in (1000, 3000, 10000, 30000, 100000, 300000, 1000000) if t <= maxn
]:
    addt = 0.0
    addn = 0
    while n < tgt:
        x, y = gen(1000, r)
        t = time.perf_counter()
        m.add(x, y)
        m.schedule_background_flush()
        addt += time.perf_counter() - t
        addn += 1000
        n += 1000
    t = time.perf_counter()
    m.ensure_index_sync()
    sync = time.perf_counter() - t
    q, _ = gen(100, qr)
    m.posterior(q[:5], params=P)
    _rust.take_walk_totals()
    ts = []
    for i in range(100):
        t = time.perf_counter()
        m.posterior(q[i : i + 1], params=P)
        ts.append(time.perf_counter() - t)
    tot = _rust.take_walk_totals()
    print(
        json.dumps(
            {
                "seed": seed,
                "N": n,
                "med_ms": float(np.median(ts)) * 1e3,
                "leaves_per_q": tot[0] / 100,
                "add_us_per_row": addt / addn * 1e6,
                "sync_s": sync,
                "rss_anon_mb": anon(),
                "lnN": float(np.log(n)),
            }
        ),
        flush=True,
    )
