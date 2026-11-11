import json
import os
import sys
import time

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


def gen(n, r):
    z = r.standard_normal((n, 4))
    x = z @ A + 0.1 * r.standard_normal((n, D))
    return x, (np.sin(z[:, 0]) + 0.1 * r.standard_normal(n)).reshape(-1, 1)


r = np.random.default_rng(0)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=sys.argv[1])
m.add(np.empty((0, D)), np.empty((0, 1)))
n = 0
qr = np.random.default_rng(5)
for tgt in (1000, 3000, 10000, 30000, 100000, 300000):
    while n < tgt:
        x, y = gen(1000, r)
        m.add(x, y)
        m.schedule_background_flush()
        n += 1000
    m.ensure_index_sync()
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
                "N": n,
                "med_ms": float(np.median(ts)) * 1e3,
                "mean_ms": float(np.mean(ts)) * 1e3,
                "leaves_f32_per_q": tot[0] / 100,
                "walks_f32_per_q": tot[1] / 100,
                "f64_leaves_per_q": tot[3] / 100,
                "lnN": float(np.log(n)),
            }
        ),
        flush=True,
    )
