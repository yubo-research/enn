import json
import os
import resource
import shutil
import sys
import tempfile
import time

import numpy as np

CACHE = sys.argv[4] if len(sys.argv) > 4 else "128000000"
os.environ.setdefault("ENN_RESIDENT_BUDGET_BYTES", CACHE)
from enn.enn.enn_class import EpistemicNearestNeighbors as ENN
from enn.enn.enn_params import ENNParams

MAX_EXP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
FIT_MIN, NQ, CHUNK = 100_000, 200, 100_000
work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="scaling_check_")
nmax = int(float(sys.argv[2])) if len(sys.argv) > 2 else 1_000_000
D = int(sys.argv[3]) if len(sys.argv) > 3 else 16
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
)
w = np.random.default_rng(0).normal(size=D) / np.sqrt(D)


def gen(rng, n):
    x = rng.random((n, D))
    return x, (np.sin(2 * np.pi * (x @ w)) + 0.1 * rng.normal(size=n))[:, None]


def snap():
    r = resource.getrusage(resource.RUSAGE_SELF)
    io = read_proc_io()
    return np.array(
        [
            time.perf_counter(),
            r.ru_utime + r.ru_stime,
            int(io["read_bytes"]),
            int(io["write_bytes"]),
        ]
    )


def read_proc_io():
    with open("/proc/self/io") as f:
        return dict(line.split(": ") for line in f.read().splitlines())


def timed(f, per, *args, **kw):
    a = snap()
    f(*args, **kw)
    d = snap() - a
    return {
        "wall_us": d[0] / per * 1e6,
        "cpu_us": d[1] / per * 1e6,
        "rd_B": d[2] / per,
        "wr_B": d[3] / per,
    }


shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
rng, qrng = np.random.default_rng(1), np.random.default_rng(2)
x, y = gen(rng, 1)
enn = ENN(x, y, work_dir=work)
cps = [b * 10**e for e in range(8) for b in (1, 3) if b * 10**e <= nmax]
rows = []


def add_rows(count):
    got = 0
    while got < count:
        k = min(CHUNK, count - got)
        xx, yy = gen(rng, k)
        enn.add(xx, yy)
        got += k


for cp in cps:
    n0 = len(enn)
    if cp <= n0:
        continue
    xq = qrng.random((NQ, D))
    r = {"n": cp, "add": timed(add_rows, cp - n0, cp - n0)}
    r["post"] = timed(enn.posterior, NQ, xq, params=P)
    r["draw"] = timed(
        enn.posterior_function_draw, NQ, xq, params=P, function_seeds=np.arange(NQ)
    )
    rows.append(r)
    print(
        f"N={cp:>8} "
        + " ".join(
            f"{op}:cpu={r[op]['cpu_us']:.1f}us wall={r[op]['wall_us']:.1f}us rd={r[op]['rd_B']:.0f}B wr={r[op]['wr_B']:.0f}B"
            for op in ("add", "post", "draw")
        ),
        flush=True,
    )
fit = [r for r in rows if r["n"] >= FIT_MIN]
bad = False
if len(fit) >= 3:
    ln = np.log([r["n"] for r in fit])
    for op, key in [
        ("add", "cpu_us"),
        ("post", "cpu_us"),
        ("draw", "cpu_us"),
        ("add", "wr_B"),
        ("post", "rd_B"),
    ]:
        v = np.array([max(r[op][key], 1e-9) for r in fit])
        if key == "rd_B" and v.max() < 4096:
            continue
        p = np.polyfit(ln, np.log(v), 1)[0]
        ok = p <= MAX_EXP
        bad |= not ok
        print(
            f"fit {op:<5}{key:<7} exponent p={p:+.2f}  {'ok' if ok else f'FAIL (> {MAX_EXP:.2f})'}"
        )
out = os.environ.get("SCALING_CHECK_OUT")
if out:
    with open(out, "w") as f:
        json.dump(rows, f)
shutil.rmtree(work, ignore_errors=True)
sys.exit(1 if bad else 0)
