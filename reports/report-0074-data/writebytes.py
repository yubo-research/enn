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


work, max_n = sys.argv[1], int(sys.argv[2])
D = 1000
B = 1000
shutil.rmtree(work, ignore_errors=True)
rng = np.random.default_rng(0)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=work)
m.add(np.empty((0, D)), np.empty((0, 1)))
n, marks = 0, {1000, 3000, 10000, 30000, 100000, 300000, 1000000}
i0, t0, n0 = io(), time.perf_counter(), 0
while n < max_n:
    m.add(rng.standard_normal((B, D)), rng.standard_normal((B, 1)))
    m.schedule_background_flush()
    n += B
    if n in marks:
        i1, t1 = io(), time.perf_counter()
        rows = n - n0
        print(
            json.dumps(
                {
                    "n": n,
                    "rows": rows,
                    "row_bytes": 8 * (D + 1),
                    "write_bytes_per_row": (i1["write_bytes"] - i0["write_bytes"])
                    / rows,
                    "wchar_per_row": (i1["wchar"] - i0["wchar"]) / rows,
                    "s_per_row": (t1 - t0) / rows,
                }
            ),
            flush=True,
        )
        i0, t0, n0 = i1, t1, n
m.ensure_index_sync()
i1 = io()
print(json.dumps({"sync_write_bytes": i1["write_bytes"] - i0["write_bytes"], "n": n}))
shutil.rmtree(work, ignore_errors=True)
