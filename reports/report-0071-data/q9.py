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
wd, d, nq = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.01
)
A = np.random.default_rng(7).standard_normal((d, D))
A = np.random.default_rng(7).standard_normal((d, D)) / 2


def io():
    return {k: int(v) for k, v in (l.split(": ") for l in open("/proc/self/io"))}


m = EpistemicNearestNeighbors.open(wd)
z = np.random.default_rng(99).standard_normal((nq, d))
xq = z @ A + 0.1 * np.random.default_rng(98).standard_normal((nq, D))
for rep in range(3):
    _rust.take_walk_totals()
    i0 = io()
    ts = []
    for i in range(nq):
        t = time.perf_counter()
        m.posterior(xq[i : i + 1], params=P, flags=DRAW_FLAGS)
        ts.append(time.perf_counter() - t)
    i1 = io()
    tot = _rust.take_walk_totals()
    print(
        json.dumps(
            {
                "rep": rep,
                "med_ms": float(np.median(ts)) * 1e3,
                "mean_ms": float(np.mean(ts)) * 1e3,
                "leaves_q": tot[0] / nq,
                "read_MB_q": (i1["read_bytes"] - i0["read_bytes"]) / nq / 1e6,
                "rchar_MB_q": (i1["rchar"] - i0["rchar"]) / nq / 1e6,
                "syscr_q": (i1["syscr"] - i0["syscr"]) / nq,
            }
        ),
        flush=True,
    )
