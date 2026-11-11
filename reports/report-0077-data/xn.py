import json
import os
import shutil
import sys
import time

os.environ["RAYON_NUM_THREADS"] = "1"
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from enn.enn.posterior_flags import PosteriorFlags

D = int(sys.argv[1])
maxn = int(sys.argv[2])
wd = "/home/dsweet/exp_tmp/wd"
shutil.rmtree(wd, ignore_errors=True)
rng = np.random.default_rng(0)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
m.add(np.empty((0, D)), np.empty((0, 1)))
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
n = 0
for t in [1000, 3000, 10000, 30000, 100000, 300000]:
    if t > maxn:
        break
    while n < t:
        b = min(1000, t - n)
        m.add(rng.standard_normal((b, D)), rng.standard_normal((b, 1)))
        n += b
    m.ensure_index_sync()
    out = {"n": t}
    for ex in (False, True):
        f = PosteriorFlags(exclude_nearest=ex)
        ts = []
        q = rng.standard_normal((1, D))
        m.posterior(q, params=P, flags=f)
        for _ in range(15):
            q = rng.standard_normal((1, D))
            t0 = time.perf_counter()
            m.posterior(q, params=P, flags=f)
            ts.append(time.perf_counter() - t0)
        out[f"ex_{ex}"] = float(np.median(ts))
    print(json.dumps(out), flush=True)
