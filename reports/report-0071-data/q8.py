import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("RAYON_NUM_THREADS", "1")
sys.path[:0] = ["/home/dsweet/projects/enn/src", "/home/dsweet/projects/enn"]
from enn import _rust
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from ops.stress import DRAW_FLAGS

D = 1000
wd, d, maxn, nq, out = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    sys.argv[5],
)
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.01
)
A = np.random.default_rng(7).standard_normal((d, D)) / 2


def gen(n, r):
    z = r.standard_normal((n, d))
    x = z @ A + 0.1 * r.standard_normal((n, D))
    f = np.sin(2 * z[:, 0]) + np.sin(2 * z[:, 1])
    return x, (f + 0.1 * r.standard_normal(n)).reshape(-1, 1)


def rb():
    return int(
        next(l for l in open("/proc/self/io") if l.startswith("read_bytes")).split()[1]
    )


def drop():
    for root, _, fs in os.walk(wd):
        for f in fs:
            fd = os.open(os.path.join(root, f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


os.makedirs(wd, exist_ok=True)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
m.add(np.empty((0, D)), np.empty((0, 1)))
r = np.random.default_rng(11)
xq, yq = gen(nq, np.random.default_rng(1234))
yy = yq.ravel()
cps = []
v = 100
while v <= maxn:
    cps += [v, 3 * v]
    v *= 10
cps = [c for c in cps if c <= maxn]
n = 0
t0 = time.time()
tadd = 0.0
nadd = 0
for cp in cps:
    while n < cp:
        b = min(1000, cp - n)
        x, y = gen(b, r)
        ta = time.perf_counter()
        m.add(x, y)
        m.schedule_background_flush()
        tadd += time.perf_counter() - ta
        nadd += b
        n += b
    ta = time.perf_counter()
    m.ensure_index_sync()
    tsync = time.perf_counter() - ta
    time.sleep(2)
    drop()
    _rust.take_walk_totals()
    r0 = rb()
    ts = []
    mu = []
    se = []
    for i in range(nq):
        t = time.perf_counter()
        o = m.posterior(xq[i : i + 1], params=P, flags=DRAW_FLAGS)
        ts.append(time.perf_counter() - t)
        mu.append(float(np.asarray(o.mu).ravel()[0]))
        se.append(float(np.asarray(o.se).ravel()[0]))
    tot = _rust.take_walk_totals()
    mu = np.array(mu)
    se = np.array(se)
    ll = -0.5 * np.log(2 * np.pi * se**2) - 0.5 * ((yy - mu) / se) ** 2
    print(
        json.dumps(
            {
                "d": d,
                "N": n,
                "loglik": float(ll.mean()),
                "loglik_se": float(ll.std(ddof=1) / np.sqrt(nq)),
                "med_ms": float(np.median(ts)) * 1e3,
                "mean_ms": float(np.mean(ts)) * 1e3,
                "leaves_q": tot[0] / nq,
                "budget_stops_q": tot[2] / nq,
                "tot": list(map(float, tot)),
                "read_MB_q": (rb() - r0) / nq / 1e6,
                "add_us_row": tadd / nadd * 1e6,
                "sync_s": tsync,
                "disk_gb": sum(
                    os.path.getsize(os.path.join(a, f))
                    for a, _, fs in os.walk(wd)
                    for f in fs
                )
                / 1e9,
                "elapsed": time.time() - t0,
                "rss_anon_mb": int(
                    next(
                        l
                        for l in Path("/proc/self/status").read_text().splitlines()
                        if l.startswith("RssAnon")
                    ).split()[1]
                )
                / 1024,
            }
        ),
        flush=True,
    )
    tadd = 0.0
    nadd = 0
