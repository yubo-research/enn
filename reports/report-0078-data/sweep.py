import json
import os
import shutil
import sys
import time

os.environ["RAYON_NUM_THREADS"] = "1"
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from enn.enn.posterior_flags import PosteriorFlags as F

D = int(sys.argv[1])
maxn = int(sys.argv[2])
wd = "/home/dsweet/exp_tmp/wd78"
shutil.rmtree(wd, ignore_errors=True)
rng = np.random.default_rng(0)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
m.add(np.empty((0, D)), np.empty((0, 1)))
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
Q = lambda n=1: rng.standard_normal((n, D))
cases = {
    "post": lambda: m.posterior(Q(), params=P),
    "post_ex": lambda: m.posterior(Q(), params=P, flags=F(exclude_nearest=True)),
    "post_obs": lambda: m.posterior(Q(), params=P, flags=F(observation_noise=True)),
    "cond": lambda: m.conditional_posterior(Q(), np.zeros((1, 1)), Q(), params=P),
    "cond_ex": lambda: m.conditional_posterior(
        Q(), np.zeros((1, 1)), Q(), params=P, flags=F(exclude_nearest=True)
    ),
    "batch3": lambda: m.batch_posterior(Q(), paramss=[P, P, P]),
    "batch_ex": lambda: m.batch_posterior(
        Q(), paramss=[P, P, P], flags=F(exclude_nearest=True)
    ),
    "neigh": lambda: m.neighbors(Q(), 10),
    "neigh_ex": lambda: m.neighbors(Q(), 10, exclude_nearest=True),
    "draw": lambda: m.posterior_function_draw(Q(), params=P, function_seeds=[1, 2]),
    "cdraw": lambda: m.conditional_posterior_function_draw(
        Q(), np.zeros((1, 1)), Q(), params=P, function_seeds=[1, 2]
    ),
}
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
    for k, f in cases.items():
        try:
            f()
            ts = []
            for _ in range(9):
                t0 = time.perf_counter()
                f()
                ts.append(time.perf_counter() - t0)
            out[k] = round(1e3 * float(np.median(ts)), 3)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            out[k] = "ERR " + str(e)[:60]
    print(json.dumps(out), flush=True)
