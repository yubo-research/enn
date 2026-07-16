"""Micro-profile: where does per-batch library time go in batch disk mode?"""

from __future__ import annotations

import shutil
import time

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.turbo.config.enn_index_driver import ENNIndexDriver

WORK_DIR = "_enn_prof"
NUM_DIM = 1000
BATCH = 1000
NUM_OBS = 30_000

shutil.rmtree(WORK_DIR, ignore_errors=True)
rng = np.random.default_rng(0)

model = EpistemicNearestNeighbors(
    np.empty((0, NUM_DIM)),
    np.empty((0, 1)),
    scale_x=False,
    index_driver=ENNIndexDriver.BPANN_DISK,
    work_dir=WORK_DIR,
    enn_storage="disk",
)

add_s = 0.0
flush_s = 0.0
gen_s = 0.0
n = 0
t_row = []
while n < NUM_OBS:
    t0 = time.perf_counter()
    x = rng.standard_normal((BATCH, NUM_DIM))
    y = rng.standard_normal((BATCH, 1))
    t1 = time.perf_counter()
    model.add(x, y)
    t2 = time.perf_counter()
    model.schedule_background_flush()
    t3 = time.perf_counter()
    gen_s += t1 - t0
    add_s += t2 - t1
    flush_s += t3 - t2
    n += BATCH
    t_row.append((n, t2 - t1, t3 - t2))

print(f"total: gen={gen_s:.3f} add={add_s:.3f} flush={flush_s:.3f}")
print("worst 10 batches by add:")
for n_i, a, f in sorted(t_row, key=lambda r: -r[1])[:10]:
    print(f"  n={n_i} add={a * 1e3:.1f}ms flush={f * 1e3:.1f}ms")
print("worst 10 batches by flush:")
for n_i, a, f in sorted(t_row, key=lambda r: -r[2])[:10]:
    print(f"  n={n_i} add={a * 1e3:.1f}ms flush={f * 1e3:.1f}ms")
