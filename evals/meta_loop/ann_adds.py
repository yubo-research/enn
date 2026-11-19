import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from cold_io import rss_anon_mib

from enn.enn.enn_class import EpistemicNearestNeighbors

D, MAXN, SEED, WORK, OUT = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    int(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
)
BATCH = int(os.environ.get("BATCH", "1000"))
WEIGHTING = bool(int(os.environ.get("WEIGHTING", "0")))
NOISE = 0.1


def f(x):
    return np.sin(3 * x[:, : min(5, x.shape[1])]).sum(1, keepdims=True)


def counters(m):
    met = m.metric
    if met is None:
        return {}
    s = met._snapshot()
    return {
        "refits": s.num_refits,
        "rescales": s.num_rescales,
        "rebuilds": s.num_rebuilds,
        "uses_learned": bool(s.uses_learned_metric),
    }


def add_batches(m, rng, n, checkpoint, last):
    times, events = [], []
    while n < checkpoint:
        b = min(BATCH, checkpoint - n)
        x = rng.random((b, D))
        y = f(x) + NOISE * rng.standard_normal((b, 1))
        t = time.perf_counter()
        m.add(x, y)
        dt = time.perf_counter() - t
        times.append(dt)
        n += b
        current = counters(m)
        if current != last:
            events.append({"n": n, "batch_s": dt, **current})
            last = current
    return times, events, last


def checkpoint_record(m, checkpoint, rows, times, events, started):
    t = time.perf_counter()
    m.ensure_index_sync()
    t_sync = time.perf_counter() - t
    a = np.asarray(times) if times else np.zeros(1)
    return {
        "D": D,
        "n": checkpoint,
        "batch": BATCH,
        "weighting": WEIGHTING,
        "add_us_row": 1e6 * (a.sum() + t_sync) / max(rows, 1),
        "add_us_row_med_batch": 1e6 * float(np.median(a)) / BATCH,
        "add_us_row_max_batch": 1e6 * float(a.max()) / BATCH,
        "sync_s": t_sync,
        "events": events,
        **counters(m),
        "rss_anon_mib": rss_anon_mib(),
        "store_bytes": sum(
            p.stat().st_size for p in Path(WORK).rglob("*") if p.is_file()
        ),
        "wall_s": time.perf_counter() - started,
    }


def main():
    grid = sorted({m * 10**e for e in range(10) for m in (1, 3) if m * 10**e <= MAXN})
    rng = np.random.default_rng(SEED)
    x0 = rng.random((1, D))
    y0 = f(x0) + NOISE * rng.standard_normal((1, 1))
    m = EpistemicNearestNeighbors(x0, y0, work_dir=WORK, weighting=WEIGHTING)
    n = 1
    t0 = time.perf_counter()
    last = counters(m)
    for cp in grid:
        times, events, last = add_batches(m, rng, n, cp, last)
        rec = checkpoint_record(m, cp, cp - n, times, events, t0)
        n = cp
        print(json.dumps(rec), flush=True)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
