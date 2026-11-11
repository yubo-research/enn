import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams

wd, NQ = sys.argv[1], int(sys.argv[2])
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)


def io():
    r = resource.getrusage(resource.RUSAGE_SELF)
    rb = 0
    for l in Path("/proc/self/io").read_text().splitlines():
        if l.startswith("read_bytes"):
            rb = int(l.split()[1])
    return r.ru_minflt, r.ru_majflt, rb


for r, _, fs in os.walk(wd):
    for f in fs:
        fd = os.open(os.path.join(r, f), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
a = io()
t0 = time.perf_counter()
m = EpistemicNearestNeighbors.open(wd)
b = io()
out = {
    "open_s": time.perf_counter() - t0,
    "open_majflt": b[1] - a[1],
    "open_GB": (b[2] - a[2]) / 2**30,
}
qr = np.random.default_rng(999)
rows = []
for i in range(NQ):
    q = qr.standard_normal((1, 1000))
    a = io()
    t0 = time.perf_counter()
    m.posterior(q, params=P)
    dt = time.perf_counter() - t0
    b = io()
    rows.append((dt, b[1] - a[1], b[2] - a[2]))
A = np.array(rows)
h = max(1, NQ // 2)
for name, s in (("first", A[:h]), ("last", A[h:])):
    out[name] = {
        "ms_med": 1e3 * float(np.median(s[:, 0])),
        "majflt": float(np.median(s[:, 1])),
        "KB": float(np.median(s[:, 2])) / 1024,
    }
print(json.dumps(out))
