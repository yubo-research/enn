import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors

D, N0, M, WORK, OUT = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    int(float(sys.argv[3])),
    sys.argv[4],
    sys.argv[5],
)
rng = np.random.default_rng(int(os.environ.get("SEED", "0")))


def make(n):
    x = rng.random((n, D))
    return x, np.sin(3 * x[:, :5]).sum(1, keepdims=True)


m = EpistemicNearestNeighbors(*make(1), work_dir=WORK)
n = 1
while n < N0:
    b = min(1000, N0 - n)
    m.add(*make(b))
    n += b
xs, ys = make(M)
t = time.perf_counter()
for i in range(M):
    m.add(xs[i : i + 1], ys[i : i + 1])
dt = time.perf_counter() - t
o = {
    "arm": os.environ.get("ARM", ""),
    "seed": int(os.environ.get("SEED", "0")),
    "D": D,
    "n0": N0,
    "m": M,
    "single_add_us": 1e6 * dt / M,
}
print(json.dumps(o))
with open(OUT, "a") as fh:
    fh.write(json.dumps(o) + "\n")
