import os
import sys
import time

import numpy as np

os.environ.setdefault("RAYON_NUM_THREADS", "1")
sys.path.insert(0, "/home/dsweet/projects/enn/ops")
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams

P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
N = int(sys.argv[1])
D = 1000
r = np.random.default_rng(0)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir="w")
m.add(np.empty((0, D)), np.empty((0, 1)))
for i in range(0, N, 1000):
    m.add(r.standard_normal((1000, D)), r.standard_normal((1000, 1)))
t = time.time()
m.ensure_index_sync()
print("sync", time.time() - t)
q = r.standard_normal((60, 1, D))
m.posterior(q[0], params=P)


def med(f):
    ts = []
    for x in q:
        t = time.perf_counter()
        f(x)
        ts.append(time.perf_counter() - t)
    return np.median(ts) * 1e3


print("fresh ms", med(lambda x: m.posterior(x, params=P)))
print("repeat same ms", med(lambda x: m.posterior(q[0], params=P)))
