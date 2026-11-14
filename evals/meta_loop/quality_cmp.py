import json
import os
import resource
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from scipy.stats import norm

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, N, SEED = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
WORK, OUT = sys.argv[4], sys.argv[5]
NOISE, NTEST, BATCH = 0.1, 1000, 1000
rng = np.random.default_rng(SEED)


def f(x):
    return np.sin(3 * x[:, : min(5, D)]).sum(1, keepdims=True)


x0 = rng.random((1, D))
m = EpistemicNearestNeighbors(
    x0, f(x0) + NOISE * rng.standard_normal((1, 1)), work_dir=WORK
)
t_add, n = 0.0, 1
while n < N:
    b = min(BATCH, N - n)
    x = rng.random((b, D))
    y = f(x) + NOISE * rng.standard_normal((b, 1))
    t = time.perf_counter()
    m.add(x, y)
    t_add += time.perf_counter() - t
    n += b
t = time.perf_counter()
m.ensure_index_sync()
t_sync = time.perf_counter() - t
xt = np.random.default_rng(10_000 + SEED).random((NTEST, D))
ft = f(xt)
yt = ft + NOISE * np.random.default_rng(20_000 + SEED).standard_normal((NTEST, 1))
rec = {"D": D, "n": N, "seed": SEED, "add_us_row": 1e6 * (t_add + t_sync) / N}
for k in (10,):
    P = ENNParams(
        k_num_neighbors=k,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=NOISE**2,
    )
    t = time.perf_counter()
    post = m.posterior(xt, params=P, flags=PosteriorFlags(observation_noise=True))
    rec["query_s"] = time.perf_counter() - t
    mu = np.asarray(post.mu).ravel()
    se = np.asarray(post.se).ravel()
    rec["nrmse"] = float(np.sqrt(np.mean((mu - ft.ravel()) ** 2)) / ft.std())
    rec["loglik"] = float(norm.logpdf(yt.ravel(), mu, se).mean())
    rec["mean_se"] = float(se.mean())
rec["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
with open(OUT, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(rec)
