import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors


def io():
    d = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        k, _, v = line.partition(":")
        d[k] = int(v)
    return d


def alloc(w):
    out = {}
    for dp, _, fs in os.walk(w):
        for f in fs:
            s = os.stat(os.path.join(dp, f))
            out[f] = s.st_blocks * 512
    return out


work, max_n, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
D = 1000
B = 1000
shutil.rmtree(work, ignore_errors=True)
rng = np.random.default_rng(0)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=work)
m.add(np.empty((0, D)), np.empty((0, 1)))
n, marks = 0, {10000, 30000, 100000, 200000, 300000, 600000, 1000000}
i0, t0, n0 = io(), time.perf_counter(), 0
while n < max_n:
    if mode == "iid":
        x = rng.standard_normal((B, D))
    else:
        x = rng.standard_normal(D) + 0.1 * rng.standard_normal((B, D))
    m.add(x, rng.standard_normal((B, 1)))
    m.schedule_background_flush()
    n += B
    if n in marks:
        i1, t1 = io(), time.perf_counter()
        rows = n - n0
        a = alloc(work)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "n": n,
                    "rows": rows,
                    "write_bytes_per_row": (i1["write_bytes"] - i0["write_bytes"])
                    / rows,
                    "s_per_row": (t1 - t0) / rows,
                    "alloc_per_row": {k: v / n for k, v in a.items() if v > 100000},
                }
            ),
            flush=True,
        )
        i0, t0, n0 = io(), time.perf_counter(), n
shutil.rmtree(work, ignore_errors=True)
