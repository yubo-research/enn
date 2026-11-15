import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, N, WORK, OUT = int(sys.argv[1]), int(float(sys.argv[2])), sys.argv[3], sys.argv[4]
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
F = PosteriorFlags(observation_noise=False)
rng = np.random.default_rng(int(os.environ.get("SEED", "0")))


def make(n):
    x = rng.random((n, D))
    return x, np.sin(3 * x[:, : min(5, D)]).sum(
        1, keepdims=True
    ) + 0.1 * rng.standard_normal((n, 1))


m = EpistemicNearestNeighbors(*make(1), work_dir=WORK)
ta = 0.0
for _ in range(N // 1000):
    x, y = make(1000)
    t = time.perf_counter()
    m.add(x, y)
    ta += time.perf_counter() - t
m.ensure_index_sync()
q = rng.random((200, D))
for i in range(50):
    m.posterior(q[i : i + 1], params=P, flags=F)
tq = []
for i in range(200):
    t = time.perf_counter()
    m.posterior(q[i : i + 1], params=P, flags=F)
    tq.append(time.perf_counter() - t)
o = {
    "D": D,
    "n": N,
    "advise": os.environ.get("ENN_DATA_ADVISE", "random"),
    "seed": int(os.environ.get("SEED", "0")),
    "add_us_row": 1e6 * ta / N,
    "q_us_med": 1e6 * float(np.median(tq)),
    "q_us_mean": 1e6 * float(np.mean(tq)),
}
print(json.dumps(o))
with open(OUT, "a") as fh:
    fh.write(json.dumps(o) + "\n")
