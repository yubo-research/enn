import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors

wd, NQ = sys.argv[1], int(sys.argv[2])
for r, _, fs in os.walk(wd):
    for f in fs:
        fd = os.open(os.path.join(r, f), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
t0 = time.perf_counter()
m = EpistemicNearestNeighbors.open(wd)
t_open = time.perf_counter() - t0
n = len(m)
Q = np.random.default_rng(999).standard_normal((NQ, 1000))
t0 = time.perf_counter()
got = np.vstack([m.neighbors(Q[i : i + 1], 10) for i in range(NQ)])
t_q = time.perf_counter() - t0
X = np.memmap(
    os.path.join(wd, "train_x.bin"), dtype=np.float64, mode="r", shape=(n, 1000)
)
best = np.full((NQ, 10), -1)
d2 = np.empty((NQ, n))
for s in range(0, n, 5000):
    B = np.asarray(X[s : s + 5000])
    d2[:, s : s + 5000] = (B * B).sum(1)[None, :] - 2 * Q @ B.T
want = np.argsort(d2, axis=1)[:, :10]
rec = np.mean([len(set(g.ravel()) & set(w)) / 10 for g, w in zip(got, want)])
print(
    json.dumps(
        {
            "n": n,
            "open_s": round(t_open, 3),
            "query_s_per": round(t_q / NQ, 4),
            "recall10": round(float(rec), 4),
        }
    )
)
