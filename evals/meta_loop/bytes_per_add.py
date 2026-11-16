import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import cold_io
import numpy as np
from capq_lib_0250 import resident

from enn.enn.enn_class import EpistemicNearestNeighbors

D, MAXN, WORK, OUT, DEV = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
BATCH = 1000
PROG_S = 30.0
pa, pn = 0.0, 0


def disk():
    return cold_io.disk(DEV)


def dsize(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 2**30


def make(rng, n):
    x = rng.random((n, D))
    signal = np.sin(3 * x[:, : min(5, D)]).sum(1, keepdims=True)
    y = signal + 0.1 * rng.standard_normal((n, 1))
    return x, y


grid = sorted({m * 10**e for e in range(10) for m in (1, 3) if m * 10**e <= MAXN})
rng = np.random.default_rng(0)
x, y = make(rng, 1)
model = EpistemicNearestNeighbors(x, y, work_dir=WORK)
pr, pw = disk()
n = 1
for cp in grid:
    rows = cp - n
    r0, w0 = disk()
    ta = 0.0
    while n < cp:
        b = min(BATCH, cp - n)
        x, y = make(rng, b)
        t = time.perf_counter()
        model.add(x, y)
        dt = time.perf_counter() - t
        ta += dt
        n += b
        pa += dt
        pn += b
        if pa > PROG_S:
            r, w = disk()
            q = {
                "D": D,
                "progress": 1,
                "n": n,
                "add_us_row": 1e6 * pa / pn,
                "rd_b_row": (r - pr) / pn,
                "wr_b_row": (w - pw) / pn,
            }
            if os.environ.get("FINCORE"):
                q["res_mib"] = resident(WORK)
            print(json.dumps(q), flush=True)
            with Path(OUT).open("a") as fh:
                fh.write(json.dumps(q) + "\n")
            pa, pn, pr, pw = 0.0, 0, r, w
    t = time.perf_counter()
    model.ensure_index_sync()
    ts = time.perf_counter() - t
    r1, w1 = disk()
    o = {
        "D": D,
        "n": cp,
        "rows": rows,
        "add_us_row": 1e6 * ta / max(rows, 1),
        "sync_us_row": 1e6 * ts / max(rows, 1),
        "rd_b_row": (r1 - r0) / max(rows, 1),
        "wr_b_row": (w1 - w0) / max(rows, 1),
        "disk_gib": dsize(WORK),
    }
    print(json.dumps(o), flush=True)
    with Path(OUT).open("a") as fh:
        fh.write(json.dumps(o) + "\n")
