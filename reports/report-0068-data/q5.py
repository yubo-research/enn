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
wd = sys.argv[1]
nq = int(sys.argv[2])


def rb():
    for l in Path("/proc/self/io").read_text().splitlines():
        if l.startswith("read_bytes"):
            return int(l.split(":")[1])


def drop():
    for root, _, fs in os.walk(wd):
        for f in fs:
            fd = os.open(os.path.join(root, f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


m = EpistemicNearestNeighbors.open(wd)
r = np.random.default_rng(1234)
z = r.standard_normal((nq, 4))
q = z @ A + 0.1 * r.standard_normal((nq, D))
ref = None
truth = np.sin(z[:, 0])
sq0 = None
for c in (30.0, 10.0, 3.0):
    _rust.set_leaf_coefficient(c)
    drop()
    for x in q[:3]:
        m.posterior(x[None], params=P)
    _rust.take_walk_totals()
    b0 = rb()
    ts = []
    idx = []
    mu = []
    for x in q:
        t = time.perf_counter()
        o = m.posterior(x[None], params=P)
        ts.append(time.perf_counter() - t)
        idx.append(set(np.asarray(o.idx).ravel().tolist()))
        mu.append(float(np.asarray(o.mu).ravel()[0]))
    b1 = rb()
    tot = _rust.take_walk_totals()
    mu = np.array(mu)
    if ref is None:
        ref = (idx, mu)
    ov = np.mean([len(a & b) / len(b) for a, b in zip(idx, ref[0])])
    sq = (mu - truth) ** 2
    if sq0 is None:
        sq0 = sq
    dd = sq - sq0
    print(
        json.dumps(
            {
                "rmse": float(np.sqrt(sq.mean())),
                "d_mse_vs_c30": float(dd.mean()),
                "d_mse_se": float(dd.std(ddof=1) / np.sqrt(nq)) if nq > 1 else 0,
                "c": c,
                "med_ms": float(np.median(ts)) * 1e3,
                "mean_ms": float(np.mean(ts)) * 1e3,
                "MB_q": (b1 - b0) / nq / 1e6,
                "leaves_q": tot[0] / nq,
                "budget_stops_q": tot[2] / nq,
                "overlap_vs_c30": ov,
                "mu_mad_vs_c30": float(np.mean(np.abs(mu - ref[1]))),
                "mu_sd": float(mu.std()),
            }
        ),
        flush=True,
    )
