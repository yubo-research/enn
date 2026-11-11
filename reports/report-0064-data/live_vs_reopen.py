import os
import shutil
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors

wd, N, D = sys.argv[1], int(sys.argv[2]), 1000
shutil.rmtree(wd, ignore_errors=True)
rng = np.random.default_rng(0)
m = EpistemicNearestNeighbors(np.empty((0, D)), np.empty((0, 1)), work_dir=wd)
m.add(np.empty((0, D)), np.empty((0, 1)))
for s in range(0, N, 1000):
    m.add(rng.standard_normal((1000, D)), rng.standard_normal((1000, 1)))
    m.schedule_background_flush()
m.ensure_index_sync()
m.persist_index_to_disk()
Q = np.random.default_rng(7).standard_normal((200, D))
live = np.vstack([m.neighbors(Q[i : i + 1], 10) for i in range(len(Q))])
del m
f = EpistemicNearestNeighbors.open(wd)
fast = np.vstack([f.neighbors(Q[i : i + 1], 10) for i in range(len(Q))])
print("stamp present:", os.path.exists(os.path.join(wd, "index", "tree_state.json")))
print("fast == live (identical ids):", np.array_equal(live, fast))
del f
os.remove(os.path.join(wd, "index", "tree_state.json"))
s = EpistemicNearestNeighbors.open(wd)
slow = np.vstack([s.neighbors(Q[i : i + 1], 10) for i in range(len(Q))])
ov = lambda a, b: np.mean([len(set(x) & set(y)) / 10 for x, y in zip(a, b)])
print(
    "slow == live:",
    np.array_equal(live, slow),
    " mean overlap slow/live:",
    round(ov(slow, live), 4),
)
