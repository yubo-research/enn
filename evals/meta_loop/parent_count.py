import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn._rust import set_count_parents, take_internal_totals, take_parent_counts
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, MAXN, WORK, OUT = int(sys.argv[1]), int(float(sys.argv[2])), sys.argv[3], sys.argv[4]
NQ = int(os.environ.get("NQ", "200"))
INTRINSIC = int(os.environ.get("INTRINSIC", "0"))
BASIS = (
    np.linalg.qr(np.random.default_rng(12345).standard_normal((D, INTRINSIC)))[0].T
    if INTRINSIC
    else None
)
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
F = PosteriorFlags(observation_noise=False)


def xs(g, n):
    if not INTRINSIC:
        return g.random((n, D))
    return g.random((n, INTRINSIC)) @ BASIS + 0.01 * g.standard_normal((n, D))


def rows(g, n):
    x = xs(g, n)
    z = x @ BASIS.T if INTRINSIC else x
    return x, np.sin(3 * z[:, : min(5, z.shape[1])]).sum(1, keepdims=True)


def measure(m, qx):
    take_parent_counts()
    take_internal_totals()
    per = []
    for i in range(NQ):
        m.posterior(qx[i : i + 1], params=P, flags=F)
        w, par, lv, lpx = take_parent_counts()
        ip, _ = take_internal_totals()
        per.append((w, par, lv, lpx, ip))
    a = np.array(per, dtype=float)
    keys = ("walks", "parents", "leaves", "leaf_parent_expands", "internal_expands")
    return {
        k: [float(np.mean(a[:, j])), float(np.percentile(a[:, j], 90))]
        for j, k in enumerate(keys)
    }


def main():
    set_count_parents(True)
    grid = sorted({m * 10**e for e in range(10) for m in (1, 3) if m * 10**e <= MAXN})
    g = np.random.default_rng(0)
    m = EpistemicNearestNeighbors(*rows(g, 1), work_dir=WORK)
    n = 1
    for cp in grid:
        while n < cp:
            b = min(10000, cp - n)
            m.add(*rows(g, b))
            n += b
        m.ensure_index_sync()
        o = {"D": D, "n": cp, "intrinsic": INTRINSIC, **measure(m, xs(g, NQ))}
        print(json.dumps(o), flush=True)
        with Path(OUT).open("a") as fh:
            fh.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
