import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, NQ, K, CHUNK = 100, 1000, 10, 100_000
work, out = sys.argv[1], sys.argv[2]
max_n = int(float(sys.argv[3])) if len(sys.argv) > 3 else 10**9
PARAMS = ENNParams(
    k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
FLAGS = PosteriorFlags(observation_noise=False)


def checkpoints(m):
    grid = [b * 10**e for e in range(10) for b in (1, 3)]
    return [g for g in grid if g <= m]


def make(rng, n):
    x = rng.random((n, D))
    y = np.sin(3 * x[:, :5]).sum(1, keepdims=True) + 0.1 * rng.standard_normal((n, 1))
    return x, y


def append(path, obj):
    with Path(path).open("a") as f:
        f.write(json.dumps(obj) + "\n")


def rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024


def dir_gib(p):
    return (
        sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs)
        / 2**30
    )


rng = np.random.default_rng(0)
xq = np.random.default_rng(1).random((NQ, D))
model, n, last, t_add_total = None, 0, 1, 0.0
DISK_CAP_GIB = 700.0
for cp in checkpoints(max_n):
    if model is not None and dir_gib(work) * cp / max(n, 1) > DISK_CAP_GIB:
        msg = {"stopped": "disk", "next_n": cp, "n": n, "disk_gib": dir_gib(work)}
        print(json.dumps(msg), flush=True)
        append(out, msg)
        break
    t_add, n_add = 0.0, 0
    while n < cp:
        b = min(max(last, 1), cp - n)
        for s in range(0, b, CHUNK):
            x, y = make(rng, min(CHUNK, b - s))
            t0 = time.perf_counter()
            if model is None:
                model = EpistemicNearestNeighbors(x, y, work_dir=work)
            else:
                model.add(x, y)
            model.ensure_index_sync()
            t_add += time.perf_counter() - t0
            n_add += len(x)
        n += b
        last = cp if n >= cp else last
    t0 = time.perf_counter()
    post = model.posterior(xq, params=PARAMS, flags=FLAGS)
    t_post = time.perf_counter() - t0
    t0 = time.perf_counter()
    draws, _ = model.posterior_function_draw(
        xq, params=PARAMS, function_seeds=[0], flags=FLAGS
    )
    t_draw = time.perf_counter() - t0
    row = {
        "n": cp,
        "n_len": len(model),
        "n_added": n_add,
        "add_s": t_add,
        "add_us": 1e6 * t_add / max(n_add, 1),
        "post_s": t_post,
        "draw_s": t_draw,
        "rss_mib": rss_mib(),
        "disk_gib": dir_gib(work),
        "index_mib": model.index_memory_bytes() / 2**20,
        "finite": bool(np.isfinite(post.mu).all() and np.isfinite(draws).all()),
    }
    print(json.dumps(row), flush=True)
    append(out, row)
    last = cp
