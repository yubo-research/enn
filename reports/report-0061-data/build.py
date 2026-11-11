import os
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors

wd, N, D = sys.argv[1], int(sys.argv[2]), 1000
rng = np.random.default_rng(0)
m = EpistemicNearestNeighbors(
    np.empty((0, D)),
    np.empty((0, 1)),
    work_dir=wd,
    **({"aggregate_duplicates": True} if os.environ.get("AGG") else {}),
)
m.add(np.empty((0, D)), np.empty((0, 1)))
n = 0
while n < N:
    b = min(1000, N - n)
    m.add(rng.standard_normal((b, D)), rng.standard_normal((b, 1)))
    m.schedule_background_flush()
    n += b
m.ensure_index_sync()
m.persist_index_to_disk()
print("built", n)
