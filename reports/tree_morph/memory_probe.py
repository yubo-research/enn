"""Resident memory of BPANN_DISK+AUTO while streaming up to N_MAX rows of the 12d stream past the tree's budget.

Usage: PYTHONPATH=src:. python reports/tree_morph/memory_probe.py N_MAX > OUT

Rows are generated one 1,000-row batch at a time (add + ensure_index_sync), so the harness holds no data set. At each
checkpoint it prints resident anonymous, file-backed and shared memory above the pre-model baseline, the peak resident
memory (VmHWM), the cumulative add time, the AUTO event counts so far (refits, rescales, re-partitions), the query time per point on 200 test points, and the four mappings with the
most resident memory.
"""

import sys
import tempfile
import time

import numpy as np

from evals import metric_12d
from ops.stress import DRAW_FLAGS, STRESS_PARAMS

BATCH = 1000
CHECKPOINTS = (
    250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 6_000_000, 8_000_000,
    10_000_000, 11_000_000, 12_000_000, 13_000_000, 14_000_000, 15_000_000, 16_000_000,
)


def status_kb():
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("RssAnon", "RssFile", "RssShmem", "VmHWM", "VmRSS"):
                out[k] = int(v.split()[0])
    return out


def smaps_rss_kb(lines):
    """Resident kB per mapping name, from the lines of /proc/self/smaps."""
    out, name = {}, None
    for parts in (line.split() for line in lines):
        if "-" in parts[0] and len(parts) >= 5:
            name = (parts[5] if len(parts) > 5 else "[anon]").rsplit("/", 1)[-1][:28]
        elif parts[0] == "Rss:":
            out[name] = out.get(name, 0) + int(parts[1])
    return out


def rss_by_file():
    """Resident MB per mapped file (top 4), from /proc/self/smaps."""
    with open("/proc/self/smaps") as f:
        out = smaps_rss_kb(f)
    top = sorted(out.items(), key=lambda kv: -kv[1])[:4]
    return " ".join(f"{k}:{v / 1024:.0f}" for k, v in top)


def main():
    n_max = int(sys.argv[1])
    rng = np.random.default_rng(0)
    x_test, _ = metric_12d.make_data(200, np.random.default_rng(7))
    base = status_kb()
    with tempfile.TemporaryDirectory(prefix="enn_mem_", dir="/tmp") as work_dir:
        x, y = metric_12d.make_data(BATCH, rng)
        model = metric_12d.build_model("bpann_disk_auto", x, y, work_dir)
        model.ensure_index_sync()
        n, t0 = BATCH, time.perf_counter()
        while n < n_max:
            x, y = metric_12d.make_data(BATCH, rng)
            model.add(x, y)
            model.ensure_index_sync()
            n += BATCH
            if n in CHECKPOINTS:
                add_s = time.perf_counter() - t0
                tq = time.perf_counter()
                model.posterior(x_test, params=STRESS_PARAMS, flags=DRAW_FLAGS)
                q_ms = 1e3 * (time.perf_counter() - tq) / len(x_test)
                s = status_kb()
                d = {k: s[k] - base.get(k, 0) for k in s}
                m = model.metric
                print(
                    f"n={n} anon_mb={d['RssAnon'] / 1024:.1f} file_mb={d['RssFile'] / 1024:.1f} "
                    f"shmem_mb={d['RssShmem'] / 1024:.1f} hwm_mb={s['VmHWM'] / 1024:.1f} "
                    f"add_s_cum={add_s:.1f} refits={m.num_refits} rescales={m.num_rescales} "
                    f"repartitions={m.num_rebuilds} query_ms={q_ms:.3f} top={rss_by_file()}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
