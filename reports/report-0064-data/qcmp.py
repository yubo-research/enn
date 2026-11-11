import json
import os
import resource
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams

wd, NQ, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)


def io():
    r = resource.getrusage(resource.RUSAGE_SELF)
    rb = 0
    with open("/proc/self/io") as f:
        for l in f:
            if l.startswith("read_bytes"):
                rb = int(l.split()[1])
    return r.ru_majflt, rb


for r, _, fs in os.walk(wd):
    for f in fs:
        fd = os.open(os.path.join(r, f), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
a = io()
t0 = time.perf_counter()
m = EpistemicNearestNeighbors.open(wd)
b = io()
res = {
    "open_s": round(time.perf_counter() - t0, 3),
    "open_GB": round((b[1] - a[1]) / 2**30, 3),
}
qr = np.random.default_rng(999)
ts = []
mus = []
ses = []
a = io()
t_all = time.perf_counter()
for i in range(NQ):
    q = qr.standard_normal((1, 1000))
    t0 = time.perf_counter()
    p = m.posterior(q, params=P)
    ts.append(time.perf_counter() - t0)
    mus.append(np.asarray(p.mu))
    ses.append(np.asarray(p.se))
b = io()
res.update(
    first_q_s=round(ts[0], 3),
    med_q_ms=round(1e3 * float(np.median(ts)), 2),
    total_q_s=round(time.perf_counter() - t_all, 2),
    query_GB=round((b[1] - a[1]) / 2**30, 3),
    query_majflt=b[0] - a[0],
)
np.save(out_path, np.concatenate([np.concatenate(mus), np.concatenate(ses)]))
print(json.dumps(res))
