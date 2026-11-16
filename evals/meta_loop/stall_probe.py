import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors

try:
    from enn._rust import take_pbuf_fault_split
except ImportError:
    take_pbuf_fault_split = None

D, MAXN, WORK, OUT, DEV = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
WINDOW = int(float(os.environ.get("WINDOW", "1e5")))
GRID = [
    g for g in (1e4, 1e5, 3e5, 1e6, 2e6, 3e6, 5e6, 7e6, 1e7, 2e7) if g + WINDOW <= MAXN
]
CG = Path(
    "/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text().split(":")[2].strip()
)


def pressure(name):
    out = {}
    for l in (CG / name).read_text().splitlines():
        k, *f = l.split()
        out[k] = int(dict(x.split("=") for x in f)["total"])
    return out


def memstat():
    s = dict(l.split() for l in (CG / "memory.stat").read_text().splitlines())
    return (
        int(s["file_dirty"]) / 2**20,
        int(s["file_writeback"]) / 2**20,
        int(s["pgmajfault"]),
    )


def dstat():
    f = next(
        l.split()
        for l in Path("/proc/diskstats").read_text().splitlines()
        if l.split()[2] == DEV
    )
    return int(f[3]), int(f[5]) * 512, int(f[7]), int(f[9]) * 512


def snap():
    t = os.times()
    io, mem = pressure("io.pressure"), pressure("memory.pressure")
    return np.array(
        [
            time.perf_counter(),
            t.user + t.system,
            io["some"],
            io["full"],
            mem["some"],
            memstat()[2],
            *dstat(),
        ]
    )


def make(rng, n):
    x = rng.random((n, D))
    return x, np.sin(3 * x[:, : min(5, D)]).sum(1, keepdims=True)


def main():
    rng = np.random.default_rng(0)
    m = EpistemicNearestNeighbors(*make(rng, 1), work_dir=WORK)
    n = 1
    for g in GRID:
        while n < g:
            b = int(min(1000, g - n))
            m.add(*make(rng, b))
            n += b
        if take_pbuf_fault_split:
            take_pbuf_fault_split()
        s0, k = snap(), 0
        while k < WINDOW:
            m.add(*make(rng, 1000))
            k += 1000
        s1 = snap()
        n += k
        d = s1 - s0
        dirty, wb, _ = memstat()
        split = take_pbuf_fault_split() if take_pbuf_fault_split else (0, 0, 0)
        o = {
            "D": D,
            "n": n,
            "wall_us_row": 1e6 * d[0] / k,
            "cpu_us_row": 1e6 * d[1] / k,
            "io_some_us_row": d[2] / k,
            "io_full_us_row": d[3] / k,
            "mem_some_us_row": d[4] / k,
            "majflt_row": d[5] / k,
            "rd_ops_row": d[6] / k,
            "rd_b_row": d[7] / k,
            "wr_ops_row": d[8] / k,
            "wr_b_row": d[9] / k,
            "dirty_mib": dirty,
            "writeback_mib": wb,
            "flt_take_row": split[0] / k,
            "flt_push_row": split[1] / k,
            "flt_leaf_row": split[2] / k,
        }
        print(json.dumps(o), flush=True)
        with Path(OUT).open("a") as fh:
            fh.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
