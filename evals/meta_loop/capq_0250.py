import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from capq_lib_0250 import resident
from cold_io import disk, rss_anon_mib, sine_rows

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, MAXN, WORK, OUT, DEV = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
NQ, NF = int(os.environ.get("NQ", "100")), int(os.environ.get("NF", "10"))
BATCH = 1000
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
F = PosteriorFlags(observation_noise=False)
if os.environ.get("PATIENCE"):
    import enn._rust as _r

    _r.set_patience_coefficient(float(os.environ["PATIENCE"]))


def majflt():
    return resource.getrusage(resource.RUSAGE_SELF).ru_majflt


def rd_ops():
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if f[2] == DEV:
            return int(f[3])
    return 0


def time_ops(fn, xs):
    us, rb, mf, ro = [], [], [], []
    for i in range(len(xs)):
        r0, m0, o0 = disk(DEV)[0], majflt(), rd_ops()
        t = time.perf_counter()
        fn(xs[i : i + 1])
        us.append(1e6 * (time.perf_counter() - t))
        rb.append(disk(DEV)[0] - r0)
        mf.append(majflt() - m0)
        ro.append(rd_ops() - o0)
    return us, rb, mf, ro


def summ(prefix, us, rb, mf, ro):
    a = np.asarray(us)
    return {
        f"{prefix}_us_mean": float(a.mean()),
        f"{prefix}_us_se": float(a.std(ddof=1) / np.sqrt(len(a)))
        if len(a) > 1
        else 0.0,
        f"{prefix}_us_med": float(np.median(a)),
        f"{prefix}_us_p90": float(np.percentile(a, 90)),
        f"{prefix}_rd_b": float(np.mean(rb)),
        f"{prefix}_majflt": float(np.mean(mf)),
        f"{prefix}_rd_ops": float(np.mean(ro)),
    }


def add_until(model, rng, n, cp):
    ta = 0.0
    while n < cp:
        b = min(BATCH, cp - n)
        x, y = sine_rows(rng, b, D)
        t = time.perf_counter()
        model.add(x, y)
        ta += time.perf_counter() - t
        n += b
    getattr(model, "ensure_index_sync", lambda: None)()
    return n, ta


def query_block(model, qrng):
    qx = qrng.random((NQ + NF, D))
    q = time_ops(lambda z: model.posterior(z, params=P, flags=F), qx[:NQ])
    f = time_ops(
        lambda z: model.posterior_function_draw(
            z, params=P, function_seeds=[7], flags=F
        ),
        qx[NQ:],
    )
    return {**summ("q", *q), **summ("fd", *f), "q_us": [round(v, 1) for v in q[0]]}


def emit(o):
    if os.environ.get("FINCORE"):
        o["res_mib"] = resident(WORK)
    print(json.dumps({k: v for k, v in o.items() if k != "q_us"}), flush=True)
    with Path(OUT).open("a") as fh:
        fh.write(json.dumps(o) + "\n")


def main():
    grid = sorted({m * 10**e for e in range(10) for m in (1, 3) if m * 10**e <= MAXN})
    rng = np.random.default_rng(0)
    qrng = np.random.default_rng(12345)
    x, y = sine_rows(rng, 1, D)
    model = EpistemicNearestNeighbors(x, y, work_dir=WORK)
    n = 1
    t_start = time.perf_counter()
    for cp in grid:
        rows = cp - n
        div = max(rows, 1)
        r0, w0 = disk(DEV)
        n, ta = add_until(model, rng, n, cp)
        r1, w1 = disk(DEV)
        o = {
            "D": D,
            "n": cp,
            "rows": rows,
            "add_us_row": 1e6 * ta / div,
            "rd_b_row": (r1 - r0) / div,
            "wr_b_row": (w1 - w0) / div,
            **query_block(model, qrng),
            "rss_anon_mib": rss_anon_mib(),
            "wall_s": time.perf_counter() - t_start,
        }
        emit(o)


main()
