import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from capq_lib_0250 import resident
from cold_io import disk, sine_rows

import enn._rust as R
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, N, WORK, OUT, DEV = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
VARS = sys.argv[6:]
NQ = int(os.environ.get("NQ", "200"))
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
F = PosteriorFlags(observation_noise=False)


def build():
    rng = np.random.default_rng(0)
    x, y = sine_rows(rng, 1, D)
    m = EpistemicNearestNeighbors(x, y, work_dir=WORK)
    n = 1
    while n < N:
        b = min(1000, N - n)
        x, y = sine_rows(rng, b, D)
        m.add(x, y)
        n += b
    m.ensure_index_sync()
    return m


def take_counters():
    return (
        R.take_leaf_rows() / NQ,
        R.take_internal_totals()[0] / NQ,
        R.take_exact_rows() / NQ,
        R.take_walk_totals(),
    )


def time_queries(m, qx):
    us, rb = [], []
    for i in range(NQ):
        d0 = disk(DEV)[0]
        t = time.perf_counter()
        m.posterior(qx[i : i + 1], params=P, flags=F)
        us.append(1e6 * (time.perf_counter() - t))
        rb.append(disk(DEV)[0] - d0)
    return us, rb


def run(m, v, qrng):
    R.set_code_extra(0 if v == "extra0" else 30)
    qx = qrng.random((NQ, D))
    r0 = resident(WORK)
    w0 = take_counters()[3]
    us, rb = time_queries(m, qx)
    r1 = resident(WORK)
    rows, ipages, exact, w = take_counters()
    R.set_code_extra(30)
    return {
        "variant": v,
        "q_us_mean": float(np.mean(us)),
        "q_us_se": float(np.std(us, ddof=1) / np.sqrt(NQ)),
        "q_rd_b": float(np.mean(rb)),
        "scanned_rows_per_q": rows,
        "internal_pages_per_q": ipages,
        "exact_rows_per_q": exact,
        "walks": str(w),
        "walks0": str(w0),
        "res_before_mib": r0,
        "grow_kib_per_q": {
            f: 1024 * (r1.get(f, 0) - r0.get(f, 0)) / NQ
            for f in sorted(set(r0) | set(r1))
        },
    }


def main():
    t = time.perf_counter()
    m = build()
    out = {"D": D, "N": N, "build_s": time.perf_counter() - t, "runs": []}
    qrng = np.random.default_rng(777)
    for v in VARS:
        o = run(m, v, qrng)
        print(json.dumps(o), flush=True)
        out["runs"].append(o)
    Path(OUT).write_text(json.dumps(out, indent=1))


main()
