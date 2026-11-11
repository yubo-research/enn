import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors

D = 1000
wd, start_n, win = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
shutil.rmtree(wd, ignore_errors=True)
rng = np.random.default_rng(0)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
m.add(np.empty((0, D)), np.empty((0, 1)))


def dirty():
    out, cur = {}, None
    for line in Path("/proc/self/smaps").read_text().splitlines():
        if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
            parts = line.split()
            cur = parts[5] if len(parts) > 5 else None
        elif cur and line.startswith(("Shared_Dirty", "Private_Dirty")):
            out[cur] = out.get(cur, 0) + int(line.split()[1])
    return {k: v for k, v in out.items() if k.startswith(wd)}


def add(n):
    for _ in range(n // 1000):
        m.add(rng.standard_normal((1000, D)), rng.standard_normal((1000, 1)))


add(start_n)
os.sync()
a = dirty()
add(win)
b = dirty()
for k in sorted(b):
    print(
        os.path.relpath(k, wd),
        "dirty KB per added row: %.2f" % ((b[k] - a.get(k, 0)) / win),
    )
