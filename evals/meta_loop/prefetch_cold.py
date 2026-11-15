import json
import os
import sys

import numpy as np
from cold_io import pageout, sine_rows, timed

from enn import _rust as R
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, N, WORK, OUT, DEV = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
NQ = int(os.environ.get("NQ", "30"))
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
F = PosteriorFlags(observation_noise=False)
SETTINGS = [
    (10, 0, 4),
    (0, 0, 0),
    (30, 0, 4),
    (10, 3, 4),
    (10, 10, 4),
    (10, 30, 4),
    (10, 0, 16),
    (30, 30, 16),
]


def one_setting(m, qx, rep, cp, pp, mp):
    R.set_code_prefetch(cp)
    R.set_prefetch_pages(pp)
    R.set_meta_prefetch(mp)
    us, rb = [], []
    for i in range(NQ):
        pageout(WORK)
        t, b = timed(DEV, m.posterior, qx[i : i + 1], params=P, flags=F)
        us.append(t)
        rb.append(b)
    o = {
        "D": D,
        "n": N,
        "rep": rep,
        "code_prefetch": cp,
        "prefetch_pages": pp,
        "meta_prefetch": mp,
    }
    o |= {"cold_q_us": us, "cold_q_rd_b": rb}
    print(
        json.dumps(
            {
                k: (float(np.median(v)) if isinstance(v, list) else v)
                for k, v in o.items()
            }
        ),
        flush=True,
    )
    with open(OUT, "a") as fh:
        fh.write(json.dumps(o) + "\n")


def main():
    rng = np.random.default_rng(0)
    m = EpistemicNearestNeighbors(*sine_rows(rng, 1, D), work_dir=WORK)
    for _ in range(N // 10000):
        m.add(*sine_rows(rng, 10000, D))
    m.ensure_index_sync()
    qx = rng.random((NQ, D))
    for rep in range(2):
        for cp, pp, mp in SETTINGS:
            one_setting(m, qx, rep, cp, pp, mp)


if __name__ == "__main__":
    main()
