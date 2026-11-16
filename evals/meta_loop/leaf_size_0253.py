import json
import math
import os
import sys

import numpy as np
from scipy.cluster.vq import kmeans2

D, N, SEED, OUT = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    int(sys.argv[3]),
    sys.argv[4],
)
NQ = int(os.environ.get("NQ", "100"))
K = int(os.environ.get("K", "10"))
INTRINSIC = int(os.environ.get("INTRINSIC", "0"))
CAPS = [int(c) for c in os.environ.get("CAPS", "30,100,300,1000,3000").split(",")]
ROW_B, PAGE_HDR, PAGE = 8 + D, 64, 4096
IOPS, BW, MAX_IO = 3000.0, 125 * 2**20, 256 * 1024
KEYS = (
    "L",
    "rows",
    "bytes",
    "ops",
    "t_iops_ms",
    "t_bw_ms",
    "t_model_ms",
    "true_leaves",
    "cert_L",
    "cert_rows",
    "cert_bytes",
    "cert_t_model_ms",
)
BASIS = (
    np.linalg.qr(np.random.default_rng(12345).standard_normal((D, INTRINSIC)))[0].T
    if INTRINSIC
    else None
)


def draw(g, n):
    if not INTRINSIC:
        return g.random((n, D)).astype(np.float32)
    z = g.random((n, INTRINSIC)) @ BASIS + 0.01 * g.standard_normal((n, D))
    return z.astype(np.float32)


def two_means(P, rng):
    sub = P[rng.choice(len(P), min(len(P), 4000), replace=False)]
    cen, _ = kmeans2(sub, 2, minit="++", seed=rng, iter=10)
    left = ((P - cen[0]) ** 2).sum(1) <= ((P - cen[1]) ** 2).sum(1)
    if left.all() or not left.any():
        left = np.arange(len(P)) < len(P) // 2
    return left


def split_rec(X, cap, rng):
    out, stack = [], [np.arange(len(X))]
    while stack:
        ix = stack.pop()
        if len(ix) <= cap:
            out.append(ix)
            continue
        left = two_means(X[ix], rng)
        stack += [ix[left], ix[~left]]
    return out


class Leaves:
    def __init__(self, X, cap):
        self.parts = split_rec(X, cap, np.random.default_rng(SEED))
        self.leaf = np.empty(len(X), dtype=np.int64)
        for j, ix in enumerate(self.parts):
            self.leaf[ix] = j
        self.size = np.bincount(self.leaf).astype(np.float64)
        self.cen = np.stack([X[ix].mean(0) for ix in self.parts])
        self.rad = np.array(
            [
                np.sqrt(((X[ix] - c) ** 2).sum(1).max())
                for ix, c in zip(self.parts, self.cen)
            ]
        )
        self.lb = np.ceil((PAGE_HDR + self.size * ROW_B) / PAGE) * PAGE
        self.lops = np.ceil(self.lb / MAX_IO)

    def cost(self, vis):
        byt, ops = float(self.lb[vis].sum()), float(self.lops[vis].sum())
        return byt, ops, 1e3 * max(ops / IOPS, byt / BW)


def ideal_visits(p, d2, q):
    tl = np.unique(p.leaf[np.argpartition(d2, K)[:K]])
    order = np.argsort(((p.cen - q) ** 2).sum(1), kind="stable")
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    return order[: int(pos[tl].max()) + 1], len(tl)


def certified_visits(p, d2, q):
    dc = np.sqrt(((p.cen - q) ** 2).sum(1))
    lbd = np.maximum(dc - p.rad, 0.0)
    bo = np.argsort(lbd + 1e-9 * dc, kind="stable")
    best = np.full(K, np.inf)
    n = 0
    for j in bo:
        if lbd[j] > best[-1]:
            break
        n += 1
        best = np.sort(np.concatenate([best, np.sqrt(d2[p.parts[j]])]))[:K]
    return bo[:n]


def query(p, X, q):
    d2 = ((X - q) ** 2).sum(1)
    vis, ntrue = ideal_visits(p, d2, q)
    cvis = certified_visits(p, d2, q)
    byt, ops, t = p.cost(vis)
    cbyt, _, ct = p.cost(cvis)
    return {
        "L": len(vis),
        "rows": float(p.size[vis].sum()),
        "bytes": byt,
        "ops": ops,
        "t_iops_ms": 1e3 * ops / IOPS,
        "t_bw_ms": 1e3 * byt / BW,
        "t_model_ms": t,
        "true_leaves": ntrue,
        "cert_L": len(cvis),
        "cert_rows": float(p.size[cvis].sum()),
        "cert_bytes": cbyt,
        "cert_t_model_ms": ct,
    }


def summarize(p, cap, recs):
    s = {k: float(np.median([r[k] for r in recs])) for k in KEYS}
    s.update(
        {
            "D": D,
            "N": N,
            "intrinsic": INTRINSIC,
            "cap": cap,
            "leaves": len(p.parts),
            "rows_per_leaf": float(p.size.mean()),
            "lnN": math.log(N),
            "t_model_p90_ms": float(np.percentile([r["t_model_ms"] for r in recs], 90)),
        }
    )
    return s


def main():
    X = draw(np.random.default_rng(SEED), N)
    Q = draw(np.random.default_rng(10_000 + SEED), NQ)
    for cap in CAPS:
        p = Leaves(X, cap)
        s = summarize(p, cap, [query(p, X, q) for q in Q])
        print(json.dumps(s), flush=True)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(s) + "\n")


main()
