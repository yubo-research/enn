import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_util

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
wd = sys.argv[1]
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


io = probe_util.io_counters


def drop():
    probe_util.drop_cache(wd)


def timed(m, qs):
    return probe_util.timed_queries(m, qs, P)


r = np.random.default_rng(seed)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
m.add(np.empty((0, D)), np.empty((0, 1)))
n = 0
qr = np.random.default_rng(5)
for tgt in [
    t for t in (1000, 3000, 10000, 30000, 100000, 300000, 1000000) if t <= maxn
]:
    addt = 0.0
    addn = 0
    c0 = io()
    while n < tgt:
        x, y = gen(1000, r)
        t = time.perf_counter()
        m.add(x, y)
        m.schedule_background_flush()
        addt += time.perf_counter() - t
        addn += 1000
        n += 1000
    c1 = io()
    m.ensure_index_sync()
    q, _ = gen(230, qr)
    m.posterior(q[:5], params=P)
    _rust.take_walk_totals()
    wmed, wmean, wio = timed(m, q[5:105])
    tot = _rust.take_walk_totals()
    time.sleep(2)
    os.sync()
    drop()
    cmed, cmean, cio = timed(m, q[105:135])
    _rust.take_walk_totals()
    wmed2, _, wio2 = timed(m, q[135 : 235 - 0][:95])
    print(
        json.dumps(
            {
                "seed": seed,
                "N": n,
                "warm_med_ms": wmed,
                "warm_mean_ms": wmean,
                "warm_io": wio,
                "leaves_per_q": tot[0] / 100,
                "walk_totals_per_q": [x / 100 for x in tot],
                "cold_med_ms": cmed,
                "cold_mean_ms": cmean,
                "cold_io": cio,
                "rewarm_med_ms": wmed2,
                "add_us_per_row": addt / addn * 1e6,
                "add_io_per_row": {k: (c1[k] - c0[k]) / addn for k in c0},
                "rss_anon_mb": anon(),
            }
        ),
        flush=True,
    )
