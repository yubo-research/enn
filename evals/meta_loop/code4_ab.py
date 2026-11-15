import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from cold_io import pageout, rss_anon_mib, sine_rows, timed

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

ARM, D, MAXN, WORK, OUT, DEV = (
    sys.argv[1],
    int(sys.argv[2]),
    int(float(sys.argv[3])),
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
)
NQ = int(os.environ.get("NQ", "20"))
INTRINSIC = int(os.environ.get("INTRINSIC", "0"))
BASIS = (
    np.linalg.qr(np.random.default_rng(12345).standard_normal((D, INTRINSIC)))[0].T
    if INTRINSIC
    else None
)


def draw(rng, n):
    return rng.random((n, INTRINSIC)) @ BASIS + 0.01 * rng.standard_normal((n, D))


def rows(rng, n):
    if not INTRINSIC:
        return sine_rows(rng, n, D)
    x = draw(rng, n)
    y = np.sin(3 * (x @ BASIS.T)).sum(1, keepdims=True) + 0.1 * rng.standard_normal(
        (n, 1)
    )
    return x, y


def queries(rng):
    return draw(rng, NQ) if INTRINSIC else rng.random((NQ, D))


P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
F = PosteriorFlags(observation_noise=False)


def query(m, q, cold):
    if cold:
        pageout(WORK)
    out = []
    us, rb = timed(DEV, lambda: out.append(m.posterior(q, params=P, flags=F)))
    post = out[0]
    return (
        us,
        rb,
        float(np.asarray(post.mu).ravel()[0]),
        float(np.asarray(post.se).ravel()[0]),
    )


def at_n(m, n, qx, add_us):
    o = {
        "arm": ARM,
        "code4": os.environ.get("ENN_CODE4", "0"),
        "D": D,
        "n": n,
        "intrinsic": INTRINSIC,
    }
    o.update(cold=[], rd=[], warm=[], mu=[], se=[], add_us_row=add_us)
    for i in range(NQ):
        q = qx[i : i + 1]
        us, rb, mu, se = query(m, q, True)
        o["cold"].append(us)
        o["rd"].append(rb)
        o["mu"].append(mu)
        o["se"].append(se)
        query(m, q, False)
        o["warm"].append(query(m, q, False)[0])
    o["rss_anon_mib"] = rss_anon_mib()
    o["disk_mib"] = (
        sum(f.stat().st_size for f in Path(WORK).rglob("*") if f.is_file()) / 2**20
    )
    return o


def main():
    grid = sorted({v * 10**e for e in range(8) for v in (1, 3) if v * 10**e <= MAXN})
    rng = np.random.default_rng(0)
    m = EpistemicNearestNeighbors(*rows(rng, 1), work_dir=WORK)
    n = 1
    for cp in grid:
        t = time.perf_counter()
        added = 0
        while n < cp:
            b = min(10000, cp - n)
            m.add(*rows(rng, b))
            n += b
            added += b
        m.ensure_index_sync()
        add_us = 1e6 * (time.perf_counter() - t) / added if added else None
        o = at_n(m, cp, queries(rng), add_us)
        print(
            json.dumps(
                {
                    "arm": ARM,
                    "n": cp,
                    "cold_ms": float(np.median(o["cold"])) / 1e3,
                    "warm_us": float(np.median(o["warm"])),
                    "pages": float(np.median(o["rd"])) / 4096,
                    "add_us_row": add_us,
                    "disk_mib": o["disk_mib"],
                }
            ),
            flush=True,
        )
        with Path(OUT).open("a") as fh:
            fh.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
