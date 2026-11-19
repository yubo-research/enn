import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from cold_io import rss_anon_mib
from scipy.stats import norm

import enn._rust as R
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, MAXN, SEED, WORK, OUT = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    int(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
)
K = int(os.environ.get("K", "10"))
NTEST = int(os.environ.get("NTEST", "1000"))
NLAT = int(os.environ.get("NLAT", "100"))
BATCH = int(os.environ.get("BATCH", "1000"))
INTRINSIC = int(os.environ.get("INTRINSIC", "0"))
KS = [int(v) for v in os.environ.get("KS", "").split(",") if v]
ARMS = os.environ.get("ARMS", "default").split(";")
NOISE = 0.1
P = ENNParams(
    k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=NOISE**2
)
F = PosteriorFlags(observation_noise=True)

DEFAULTS = {
    "leaf_coef": (R.set_leaf_coefficient, 0.0),
    "internal_coef": (R.set_internal_coefficient, 0.0),
    "leaf_codes": (lambda v: R.set_leaf_codes(bool(v)), 1),
    "code_extra": (lambda v: R.set_code_extra(int(v)), 30),
    "icache": (lambda v: R.set_internal_cache_bytes(int(v)), 64 << 20),
    "code_prefetch": (lambda v: R.set_code_prefetch(int(v)), 10),
    "meta_prefetch": (lambda v: R.set_meta_prefetch(int(v)), 4),
    "stop_eps": (R.set_stop_epsilon, 1.0),
    "radius_w": (R.set_radius_weight, 0.1),
    "patience": (R.set_patience_coefficient, 0.0),
    "rerank_skip": (lambda v: R.set_f32_rerank_skip(bool(v)), 1),
}


def apply(arm):
    for name, (fn, dflt) in DEFAULTS.items():
        fn(dflt)
    for kv in arm.split(","):
        if "=" in kv:
            name, v = kv.split("=")
            DEFAULTS[name][0](float(v))


BASIS = (
    np.linalg.qr(np.random.default_rng(12345).standard_normal((D, INTRINSIC)))[0].T
    if INTRINSIC
    else None
)


def f(x):
    z = x @ BASIS.T if INTRINSIC else x
    return np.sin(3 * z[:, : min(5, z.shape[1])]).sum(1, keepdims=True)


def draw(g, n):
    if not INTRINSIC:
        return g.random((n, D))
    return g.random((n, INTRINSIC)) @ BASIS + 0.01 * g.standard_normal((n, D))


def exact_knn(X, xt, k):
    out = np.empty((len(xt), k), dtype=np.int64)
    xx = (X * X).sum(1)
    for s in range(0, len(xt), 20):
        q = xt[s : s + 20]
        d2 = (q * q).sum(1)[:, None] + xx[None, :] - 2.0 * q @ X.T
        out[s : s + 20] = np.argpartition(d2, k, axis=1)[:, :k]
    return out


def drain():
    R.take_walk_totals()
    R.take_exact_rows()
    R.take_internal_cache_stats()


def store_bytes():
    return sum(p.stat().st_size for p in Path(WORK).rglob("*") if p.is_file())


def measure_neighbors(m, xt, k):
    drain()
    t = time.perf_counter()
    approx = m.neighbors(xt, k)
    t_nb = time.perf_counter() - t
    walk = R.take_walk_totals()
    exact_rows = R.take_exact_rows()
    icache = R.take_internal_cache_stats()
    t = time.perf_counter()
    m.neighbors(xt, k)
    t_nb2 = time.perf_counter() - t
    drain()
    timings = {
        "nb_ms_q": 1e3 * min(t_nb, t_nb2) / len(xt),
        "nb_ms_q_first": 1e3 * t_nb / len(xt),
        "nb_ms_q_second": 1e3 * t_nb2 / len(xt),
    }
    counters = {
        "walk": walk,
        "exact_rows_q": exact_rows / len(xt),
        "icache_hit_miss": list(icache),
    }
    return approx, timings, counters


def nrmse(prediction, target):
    return float(np.sqrt(np.mean((prediction - target.ravel()) ** 2)) / target.std())


def measure_posterior(m, xt):
    lat = []
    for i in range(min(NLAT, len(xt))):
        t = time.perf_counter()
        m.posterior(xt[i : i + 1], params=P, flags=F)
        lat.append(1e3 * (time.perf_counter() - t))
    t = time.perf_counter()
    post = m.posterior(xt, params=P, flags=F)
    t_post = time.perf_counter() - t
    mu = np.asarray(post.mu).ravel()
    se = np.asarray(post.se).ravel()
    timings = {
        "post_batch_ms_q": 1e3 * t_post / len(xt),
        "post_ms_med": float(np.median(lat)),
        "post_ms_p90": float(np.percentile(lat, 90)),
        "post_ms_mean": float(np.mean(lat)),
    }
    return mu, se, timings


def evaluate(m, y, xt, ft, yt, k, exact, *, arm):
    apply(arm)
    approx, neighbor_timings, counters = measure_neighbors(m, xt, k)
    recall = float(
        np.mean(
            [len(set(a.tolist()) & set(e.tolist())) / k for a, e in zip(approx, exact)]
        )
    )
    mu, se, posterior_timings = measure_posterior(m, xt)
    rec = {
        "arm": arm,
        "k": k,
        **neighbor_timings,
        **posterior_timings,
        "recall": recall,
        "nrmse": nrmse(mu, ft),
        "loglik": float(norm.logpdf(yt.ravel(), mu, se).mean()),
        "knn_nrmse_approx": nrmse(y[approx].mean(1), ft),
        **counters,
    }
    apply("default")
    return rec


def add_batches(m, rng, xs, ys, n, checkpoint):
    t_add = 0.0
    while n < checkpoint:
        b = min(BATCH, checkpoint - n)
        x = draw(rng, b)
        y = f(x) + NOISE * rng.standard_normal((b, 1))
        t = time.perf_counter()
        m.add(x, y)
        t_add += time.perf_counter() - t
        xs.append(x.astype(np.float32))
        ys.append(y)
        n += b
    return t_add


def checkpoint_metrics(m, checkpoint, rows, t_add, t_sync, started):
    return {
        "D": D,
        "n": checkpoint,
        "seed": SEED,
        "intrinsic": INTRINSIC,
        "batch": BATCH,
        "add_us_row": 1e6 * (t_add + t_sync) / max(rows, 1),
        "sync_us_row": 1e6 * t_sync / max(rows, 1),
        "rss_anon_mib": rss_anon_mib(),
        "store_bytes": store_bytes(),
        "index_bytes": m.index_memory_bytes(),
        "wall_s": time.perf_counter() - started,
    }


def write_checkpoint(m, base, x, y, xt, ft, yt, *, ks):
    for k in ks:
        if base["n"] <= k:
            continue
        exact = exact_knn(x, xt.astype(np.float32), k)
        for arm in ARMS:
            rec = {**base, **evaluate(m, y, xt, ft, yt, k, exact, arm=arm)}
            print(
                json.dumps({key: value for key, value in rec.items() if key != "walk"}),
                flush=True,
            )
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


def main():
    grid = sorted({m * 10**e for e in range(10) for m in (1, 3) if m * 10**e <= MAXN})
    rng = np.random.default_rng(SEED)
    xt = draw(np.random.default_rng(10_000 + SEED), NTEST)
    ft = f(xt)
    yt = ft + NOISE * np.random.default_rng(20_000 + SEED).standard_normal((NTEST, 1))
    x0 = draw(rng, 1)
    y0 = f(x0) + NOISE * rng.standard_normal((1, 1))
    m = EpistemicNearestNeighbors(x0, y0, work_dir=WORK)
    xs, ys = [x0.astype(np.float32)], [y0]
    n = 1
    t0 = time.perf_counter()
    for cp in grid:
        rows = cp - n
        t_add = add_batches(m, rng, xs, ys, n, cp)
        n = cp
        t = time.perf_counter()
        m.ensure_index_sync()
        t_sync = time.perf_counter() - t
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0).ravel()
        write_checkpoint(
            m,
            checkpoint_metrics(m, cp, rows, t_add, t_sync, t0),
            x,
            y,
            xt,
            ft,
            yt,
            ks=KS if (KS and cp == grid[-1]) else [K],
        )


if __name__ == "__main__":
    main()
