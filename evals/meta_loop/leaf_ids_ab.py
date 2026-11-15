import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from cold_io import pageout, sine_rows, timed

from enn import _rust
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, MAXN, WORK, OUT, DEV = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
NQ = int(os.environ.get("NQ", "20"))
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
F = PosteriorFlags(observation_noise=False)


def run(m, q, on, cold):
    _rust.set_leaf_ids(on)
    if cold:
        pageout(WORK)
    out = []
    us, rb = timed(DEV, lambda: out.append(m.posterior(q, params=P, flags=F)))
    return us, rb, float(np.asarray(out[0].mu).ravel()[0])


def at_n(m, n, qx):
    o = {
        "D": D,
        "n": n,
        "cold_on": [],
        "cold_off": [],
        "rd_on": [],
        "rd_off": [],
        "warm_on": [],
        "warm_off": [],
        "same_mu": True,
    }
    for i in range(NQ):
        q = qx[i : i + 1]
        order = (True, False) if i % 2 == 0 else (False, True)
        mus = {}
        for on in order:
            us, rb, mu = run(m, q, on, True)
            tag = "on" if on else "off"
            o[f"cold_{tag}"].append(us)
            o[f"rd_{tag}"].append(rb)
            mus[on] = mu
        o["same_mu"] &= mus[True] == mus[False]
        for on in order:
            run(m, q, on, False)
            o["warm_on" if on else "warm_off"].append(run(m, q, on, False)[0])
    return o


def main():
    grid = sorted({v * 10**e for e in range(8) for v in (1, 3) if v * 10**e <= MAXN})
    rng = np.random.default_rng(0)
    m = EpistemicNearestNeighbors(*sine_rows(rng, 1, D), work_dir=WORK)
    n = 1
    for cp in grid:
        while n < cp:
            b = min(10000, cp - n)
            m.add(*sine_rows(rng, b, D))
            n += b
        m.ensure_index_sync()
        o = at_n(m, cp, rng.random((NQ, D)))
        print(
            json.dumps(
                {
                    "n": cp,
                    "same": o["same_mu"],
                    **{
                        k: float(np.median(o[k])) / 1e3
                        for k in ("cold_on", "cold_off", "warm_on", "warm_off")
                    },
                    "pg_on": float(np.median(o["rd_on"])) / 4096,
                    "pg_off": float(np.median(o["rd_off"])) / 4096,
                }
            ),
            flush=True,
        )
        with Path(OUT).open("a") as fh:
            fh.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
