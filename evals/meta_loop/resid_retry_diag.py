import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
sys.argv = [
    sys.argv[0],
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
    "none",
]
import code4_ab as c

from enn import _rust
from enn.enn.enn_class import EpistemicNearestNeighbors


def main():
    import numpy as np

    grid = sorted({v * 10**e for e in range(8) for v in (1, 3) if v * 10**e <= c.MAXN})
    rng = np.random.default_rng(0)
    m = EpistemicNearestNeighbors(*c.rows(rng, 1), work_dir=c.WORK)
    n = 1
    for cp in grid:
        while n < cp:
            b = min(10000, cp - n)
            m.add(*c.rows(rng, b))
            n += b
        m.ensure_index_sync()
        qx = c.queries(rng)
        m.posterior(qx[:1], params=c.P, flags=c.F)
        _rust.take_coded_retries()
        _rust.take_exact_rows()
        t = time.perf_counter()
        for i in range(c.NQ):
            m.posterior(qx[i : i + 1], params=c.P, flags=c.F)
        us = 1e6 * (time.perf_counter() - t) / c.NQ
        o = {
            "arm": c.ARM,
            "code4": os.environ.get("ENN_CODE4", "0"),
            "n": cp,
            "intrinsic": c.INTRINSIC,
            "retries_per_q": _rust.take_coded_retries() / c.NQ,
            "exact_rows_per_q": _rust.take_exact_rows() / c.NQ,
            "warm_us": us,
        }
        print(json.dumps(o), flush=True)
        with open(c.OUT, "a") as fh:
            fh.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
