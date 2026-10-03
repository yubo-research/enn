"""Measure how BPANN_DISK+AUTO's add time, query time, RAM and disk grow with N.

Usage: PYTHONPATH=src:. python reports/tree_morph/complexity_probe.py {12d|ranges} > OUT

Streams 1e6 rows in 500-row batches (add + ensure_index_sync), seed 0. Every batch prints its wall
time and the AUTO events it triggered (F refit, S rescale, R re-partition). At each checkpoint it
prints: query time per point on 1,000 distinct test points and on one point repeated 1,000 times
(fastest of 5), anonymous RSS above the pre-model baseline (as is, and after returning freed heap to
the OS), bytes of the row files, the time of one in-place rescale of every stored coordinate
(median of 5), the time and size of a hard persist of the tree, and the page counts and
root-to-leaf depth read back from the persisted pages.bin.
"""

import ctypes
import gc
import os
import struct
import sys
import tempfile
import time

import numpy as np

from enn.enn.enn_fit import enn_fit
from evals import metric_12d, metric_ranges
from ops.stress import DRAW_FLAGS

DATA = {"12d": metric_12d.make_data, "ranges": metric_ranges.make_data}
N_MAX = 1_000_000
BATCH = 500
CHECKPOINTS = (10_000, 30_000, 100_000, 300_000, 1_000_000)
TREE_FILES = ("header.json", "pages.bin", "skip_edges.bin")


def anon_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1])
    return 0


def trimmed_anon_kb():
    """RssAnon after returning freed heap to the OS (drops the harness's transient enn_fit buffers)."""
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return anon_kb()


def file_sizes(root):
    """Bytes of the row files (and metadata) and bytes of the persisted tree files under ``root``."""
    sizes = [
        (f in TREE_FILES, os.path.getsize(os.path.join(r, f)))
        for r, _, fs in os.walk(root)
        for f in fs
    ]
    return sum(s for is_tree, s in sizes if not is_tree), sum(
        s for is_tree, s in sizes if is_tree
    )


def read_pages(root):
    """Map page id to its child page ids (internal) or its row count (leaf), from pages.bin."""
    path = next(
        os.path.join(r, "pages.bin") for r, _, fs in os.walk(root) if "pages.bin" in fs
    )
    with open(path, "rb") as f:
        data = f.read()
    (num_pages,), off = struct.unpack_from("<I", data, 0), 4
    children, rows = {}, {}
    for _ in range(num_pages):
        (length,) = struct.unpack_from("<I", data, off)
        kind, (page_id, num_dim, count, end) = (
            data[off + 8],
            struct.unpack_from("<IIII", data, off + 9),
        )
        if kind == 0:
            stride = 4 + 4 * num_dim
            children[page_id] = [
                struct.unpack_from("<I", data, off + 21 + i * stride)[0]
                for i in range(count)
            ]
        else:
            rows[page_id] = end - count if kind == 3 else count
        off += 4 + length
    return children, rows


def page_stats(root):
    """Leaves, rows in leaves, internal pages, child references and root-to-leaf depth range."""
    children, rows = read_pages(root)
    referenced = {c for cs in children.values() for c in cs}
    level, depth = [p for p in children if p not in referenced], 0
    leaf_depths = []
    while level:
        depth += 1
        leaf_depths += [depth] * sum(p in rows for p in level)
        level = [c for p in level for c in children.get(p, [])]
    return {
        "leaves": len(rows),
        "leaf_rows": sum(rows.values()),
        "internal": len(children),
        "refs": len(referenced),
        "depth_min": min(leaf_depths),
        "depth_max": max(leaf_depths),
    }


def best_ms_per_pt(model, params, xq):
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.posterior(xq, params=params, flags=DRAW_FLAGS)
        ts.append(time.perf_counter() - t0)
    return 1e3 * min(ts) / len(xq)


def rescale_ms(model):
    """Median time of one in-place rescale of every stored coordinate (then undone, untimed)."""
    scale = 1.0 / np.sqrt(model.metric.weights)
    backend = model.rust_backend
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        backend.set_metric_scale(scale * 1.01, rebuild=False)
        ts.append(time.perf_counter() - t0)
        backend.set_metric_scale(scale, rebuild=False)
    return 1e3 * float(np.median(ts))


def checkpoint(model, n, x_test, fit_rng, base_anon, work_dir):
    resc = rescale_ms(model)
    params = enn_fit(
        model, k=10, num_fit_candidates=100, num_fit_samples=100, rng=fit_rng
    )
    distinct = best_ms_per_pt(model, params, x_test)
    same = best_ms_per_pt(model, params, np.repeat(x_test[:1], len(x_test), axis=0))
    ram = anon_kb() - base_anon
    rows_bytes, _ = file_sizes(work_dir)
    t0 = time.perf_counter()
    model.persist_index_to_disk()
    persist_s = time.perf_counter() - t0
    _, tree_bytes = file_sizes(work_dir)
    ram_trim = trimmed_anon_kb() - base_anon
    pages = " ".join(f"{k}={v}" for k, v in page_stats(work_dir).items())
    print(
        f"C n={n} query_ms={distinct:.5f} query_same_ms={same:.5f} anon_kb={ram} anon_trim_kb={ram_trim} "
        f"rows_bytes={rows_bytes} tree_bytes={tree_bytes} persist_s={persist_s:.4f} rescale_ms={resc:.3f} "
        f"{pages}",
        flush=True,
    )


def counters(model):
    m = model.metric
    return (m.num_refits, m.num_rescales, m.num_rebuilds)


def stream(tag, work_dir):
    rng = np.random.default_rng(0)
    x, y = DATA[tag](N_MAX, rng)
    x_test, _ = DATA[tag](1000, rng)
    base_anon = anon_kb()
    fit_rng = np.random.default_rng(1)
    model = metric_12d.build_model("bpann_disk_auto", x[:BATCH], y[:BATCH], work_dir)
    model.ensure_index_sync()
    for start in range(BATCH, N_MAX, BATCH):
        stop = start + BATCH
        before = counters(model)
        t0 = time.perf_counter()
        model.add(x[start:stop], y[start:stop])
        model.ensure_index_sync()
        dt = time.perf_counter() - t0
        ev = "".join(c for c, a, b in zip("FSR", counters(model), before) if a > b)
        print(f"B n={stop} add_s={dt:.6f} ev={ev or '-'}", flush=True)
        if stop in CHECKPOINTS:
            checkpoint(model, stop, x_test, fit_rng, base_anon, work_dir)


def main():
    with tempfile.TemporaryDirectory(prefix="enn_complexity_") as work_dir:
        stream(sys.argv[1], work_dir)


if __name__ == "__main__":
    main()
