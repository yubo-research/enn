import json
import os
import resource
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from scipy.stats import norm

try:
    from enn._rust import leaf_coefficient, set_leaf_coefficient
except ImportError:
    leaf_coefficient = set_leaf_coefficient = None
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, N, SEED = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
LEAF = float(sys.argv[4])
WORK, OUT = sys.argv[5], sys.argv[6]
NOISE, NTEST, BATCH, K = 0.1, 1000, 1000, 10
rng = np.random.default_rng(SEED)


INTRINSIC = int(os.environ.get("INTRINSIC", "0"))
BASIS = (
    np.linalg.qr(np.random.default_rng(12345).standard_normal((D, INTRINSIC)))[0].T
    if INTRINSIC
    else None
)


def latent(x):
    return x @ BASIS.T if INTRINSIC else x


def f(x):
    z = latent(x)
    return np.sin(3 * z[:, : min(5, z.shape[1])]).sum(1, keepdims=True)


def draw(g, n):
    if not INTRINSIC:
        return g.random((n, D))
    return g.random((n, INTRINSIC)) @ BASIS + 0.01 * g.standard_normal((n, D))


if LEAF > 0:
    set_leaf_coefficient(LEAF)
for _env, _fn in (
    ("STOP_EPS", "set_stop_epsilon"),
    ("INTERNAL_COEF", "set_internal_coefficient"),
    ("RERANK_SKIP", "set_f32_rerank_skip"),
):
    if os.environ.get(_env):
        import enn._rust as _r

        _v = float(os.environ[_env])
        getattr(_r, _fn)(bool(_v) if _env == "RERANK_SKIP" else _v)
if os.environ.get("CODE_EXTRA"):
    from enn._rust import set_code_extra

    set_code_extra(int(os.environ["CODE_EXTRA"]))
x0 = draw(rng, 1)
y0 = f(x0) + NOISE * rng.standard_normal((1, 1))
m = EpistemicNearestNeighbors(x0, y0, work_dir=WORK)
X = [x0]
Y = [y0]
t_add, n = 0.0, 1
while n < N:
    b = min(BATCH, N - n)
    x = draw(rng, b)
    y = f(x) + NOISE * rng.standard_normal((b, 1))
    t = time.perf_counter()
    m.add(x, y)
    t_add += time.perf_counter() - t
    X.append(x)
    Y.append(y)
    n += b
t = time.perf_counter()
m.ensure_index_sync()
t_sync = time.perf_counter() - t
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0).ravel()

xt = draw(np.random.default_rng(10_000 + SEED), NTEST)
ft = f(xt)
yt = ft + NOISE * np.random.default_rng(20_000 + SEED).standard_normal((NTEST, 1))

exact = np.empty((NTEST, K), dtype=np.int64)
xx = (X * X).sum(1)
for s in range(0, NTEST, 20):
    q = xt[s : s + 20]
    d2 = (q * q).sum(1)[:, None] + xx[None, :] - 2.0 * q @ X.T
    exact[s : s + 20] = np.argpartition(d2, K, axis=1)[:, :K]

walk = None
try:
    from enn._rust import take_walk_totals
except ImportError:
    take_walk_totals = None
if take_walk_totals:
    take_walk_totals()
t = time.perf_counter()
approx = m.neighbors(xt, K)
t_nb = time.perf_counter() - t
if take_walk_totals:
    walk = take_walk_totals()
recall = np.mean(
    [len(set(a.tolist()) & set(e.tolist())) / K for a, e in zip(approx, exact)]
)

rmse_exact = float(np.sqrt(np.mean((Y[exact].mean(1) - ft.ravel()) ** 2)) / ft.std())
rmse_approx = float(np.sqrt(np.mean((Y[approx].mean(1) - ft.ravel()) ** 2)) / ft.std())

rec = {
    "D": D,
    "n": N,
    "seed": SEED,
    "leaf_coef": LEAF,
    "intrinsic": INTRINSIC,
    "code_extra": os.environ.get("CODE_EXTRA", "default"),
    "code4": os.environ.get("ENN_CODE4", "0"),
    "leaf_coef_read": float(leaf_coefficient()) if leaf_coefficient else -1.0,
    "add_us_row": 1e6 * (t_add + t_sync) / N,
    "recall_at_10": float(recall),
    "neighbors_ms_per_query": 1e3 * t_nb / NTEST,
    "knn_nrmse_exact": rmse_exact,
    "knn_nrmse_approx": rmse_approx,
    "stop_eps": os.environ.get("STOP_EPS", "default"),
    "internal_coef": os.environ.get("INTERNAL_COEF", "default"),
    "rerank_skip": os.environ.get("RERANK_SKIP", "default"),
    "walk_totals": walk,
}
P = ENNParams(
    k_num_neighbors=K,
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
rec["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
with open(OUT, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(rec)
