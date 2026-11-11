import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("RAYON_NUM_THREADS", "1")
sys.path.insert(0, "/home/dsweet/projects/enn/src")
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams

P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
D = 1000
A = np.random.default_rng(7).standard_normal((4, D)) / 2
wd = sys.argv[1]


def io():
    d = {}
    for l in Path("/proc/self/io").read_text().splitlines():
        k, _, v = l.partition(":")
        if k == "read_bytes":
            d[k] = int(v)
    d["majflt"] = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
    return d


for root, _, fs in os.walk(wd):
    for f in fs:
        fd = os.open(os.path.join(root, f), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
m = EpistemicNearestNeighbors.open(wd)
r = np.random.default_rng(99)
z = r.standard_normal((100, 4))
q = z @ A + 0.1 * r.standard_normal((100, D))
res = []
for part in (q[:50], q[50:]):
    c0 = io()
    ts = []
    for x in part:
        t = time.perf_counter()
        out = m.posterior(x[None], params=P)
        ts.append(time.perf_counter() - t)
    c1 = io()
    res.append(
        {
            "med_ms": float(np.median(ts)) * 1e3,
            "mean_ms": float(np.mean(ts)) * 1e3,
            "readB_q": (c1["read_bytes"] - c0["read_bytes"]) / 50,
            "majflt_q": (c1["majflt"] - c0["majflt"]) / 50,
        }
    )
print(json.dumps({"first50": res[0], "next50": res[1]}))
