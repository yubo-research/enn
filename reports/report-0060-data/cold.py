import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn._rust import take_walk_totals
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams

wd, NQ, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)


def evict():
    for r, _, fs in os.walk(wd):
        for f in fs:
            fd = os.open(os.path.join(r, f), os.O_RDONLY)
            os.fsync(fd) if False else None
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


def io():
    r = resource.getrusage(resource.RUSAGE_SELF)
    rb = 0
    for l in Path("/proc/self/io").read_text().splitlines():
        if l.startswith("read_bytes"):
            rb = int(l.split()[1])
    return r.ru_minflt, r.ru_majflt, rb


if mode in ("cold", "fresh"):
    evict()
m = EpistemicNearestNeighbors.open(wd)
qr = np.random.default_rng(999)
rows = []
for i in range(NQ):
    q = qr.standard_normal((1, 1000))
    a = io()
    take_walk_totals()
    t0 = time.perf_counter()
    m.posterior(q, params=P)
    dt = time.perf_counter() - t0
    b = io()
    rows.append((dt, b[0] - a[0], b[1] - a[1], b[2] - a[2]))
a = np.array(rows)[: max(1, NQ // 2)]
print(
    json.dumps(
        {
            "mode": mode,
            "ms_med": 1e3 * float(np.median(a[:, 0])),
            "ms_mean": 1e3 * float(a[:, 0].mean()),
            "minflt": float(np.median(a[:, 1])),
            "majflt": float(np.median(a[:, 2])),
            "KBread": float(np.median(a[:, 3])) / 1024,
            "peak_rss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
    )
)
