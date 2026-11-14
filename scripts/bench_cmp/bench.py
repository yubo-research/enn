import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D = int(sys.argv[1])
MAX_N = int(float(sys.argv[2]))
WORK = sys.argv[3]
OUT = sys.argv[4]
MAX_S = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
NQ, REPS, BATCH, RSS_STOP = 100, 5, 1000, 3300.0
QMAX_N = int(float(os.environ.get("QMAX_N", "1e18")))
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
F = PosteriorFlags(observation_noise=False)


def rss():
    d = {}
    for l in Path("/proc/self/status").read_text().splitlines():
        k, _, v = l.partition(":")
        if k in ("VmRSS", "RssAnon", "RssFile"):
            d[k] = int(v.split()[0]) / 1024
    return d


def dsize(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 2**30


def make(rng, n):
    x = rng.random((n, D))
    y = np.sin(3 * x[:, : min(5, D)]).sum(1, keepdims=True) + 0.1 * rng.standard_normal(
        (n, 1)
    )
    return x, y


def med(fn):
    fn()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(min(ts)), float(max(ts))


def emit(o):
    print(json.dumps(o), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(o) + "\n")


grid = sorted({m * 10**e for e in range(10) for m in (1, 3) if m * 10**e <= MAX_N})
rng = np.random.default_rng(0)
xq = np.random.default_rng(1).random((NQ, D))
t0all = time.perf_counter()
x, y = make(rng, 1)
model = EpistemicNearestNeighbors(x, y, work_dir=WORK)
n = 1
for cp in grid:
    rows = cp - n
    t_add = t_sync = 0.0
    while n < cp:
        b = min(BATCH, cp - n)
        x, y = make(rng, b)
        t = time.perf_counter()
        model.add(x, y)
        t_add += time.perf_counter() - t
        n += b
        if rss()["VmRSS"] > RSS_STOP:
            emit({"stopped": "rss", "n": n, **rss()})
            sys.exit(0)
    t = time.perf_counter()
    model.ensure_index_sync()
    t_sync = time.perf_counter() - t
    nan = (None, None, None)
    post = med(lambda: model.posterior(xq, params=P, flags=F)) if cp <= QMAX_N else nan
    draw = (
        med(
            lambda: model.posterior_function_draw(
                xq, params=P, function_seeds=[0], flags=F
            )
        )
        if cp <= QMAX_N
        else nan
    )
    emit(
        {
            "D": D,
            "n": cp,
            "n_len": len(model),
            "rows_added": rows,
            "add_us_row": 1e6 * t_add / max(rows, 1),
            "add_sync_us_row": 1e6 * (t_add + t_sync) / max(rows, 1),
            "sync_s": t_sync,
            "post_s": post[0],
            "post_min": post[1],
            "post_max": post[2],
            "draw_s": draw[0],
            "draw_min": draw[1],
            "draw_max": draw[2],
            "nq": NQ,
            "disk_gib": dsize(WORK),
            **rss(),
            "wall_s": time.perf_counter() - t0all,
        }
    )
    if time.perf_counter() - t0all > MAX_S:
        emit({"stopped": "time", "n": cp})
        break
