import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import cold_io
import numpy as np

from enn._rust import take_pbuf_detail, take_pbuf_stats
from enn.enn.enn_class import EpistemicNearestNeighbors

D, MAXN, WORK, OUT, DEV = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
WINDOW = int(float(os.environ.get("WINDOW", "50000")))


def majflt():
    return int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[9])


def make(rng, n):
    x = rng.random((n, D))
    return x, np.sin(3 * x[:, : min(5, D)]).sum(1, keepdims=True)


def window(m, rng):
    r0, w0 = cold_io.disk(DEV)
    f0, ta, k = majflt(), 0.0, 0
    while k < WINDOW:
        x, y = make(rng, 1000)
        t = time.perf_counter()
        m.add(x, y)
        ta += time.perf_counter() - t
        k += 1000
    r1, w1 = cold_io.disk(DEV)
    fl, fr, fns, ff = take_pbuf_stats()
    tf, lv, sp = take_pbuf_detail()
    return k, {
        "add_us_row": 1e6 * ta / k,
        "flush_us_row": 1e-3 * fns / k,
        "flushes_per_krow": 1e3 * fl / k,
        "replayed_per_row": fr / k,
        "majflt_per_row": (majflt() - f0) / k,
        "flush_majflt_per_row": ff / k,
        "take_majflt_per_flush": tf / max(fl, 1),
        "leaves_per_flush": lv / max(fl, 1),
        "splits_per_flush": sp / max(fl, 1),
        "rd_b_row": (r1 - r0) / k,
        "wr_b_row": (w1 - w0) / k,
    }


def main():
    rng = np.random.default_rng(0)
    m = EpistemicNearestNeighbors(*make(rng, 1), work_dir=WORK)
    n = 1
    take_pbuf_stats()
    while n < MAXN:
        k, stats = window(m, rng)
        n += k
        o = {
            "D": D,
            "n": n,
            "pbuf": int(os.environ.get("ENN_PBUF", "0")),
            "advise": os.environ.get("ENN_PBUF_ADVISE", "1"),
            **stats,
        }
        print(json.dumps(o), flush=True)
        with Path(OUT).open("a") as fh:
            fh.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
