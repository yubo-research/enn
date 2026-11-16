import heapq
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

from enn.enn.enn_class import EpistemicNearestNeighbors

D, N, SEED = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
WORK, OUT = sys.argv[4], sys.argv[5]
NQ = int(os.environ.get("NQ", "200"))
K = int(os.environ.get("K", "10"))
INTRINSIC = int(os.environ.get("INTRINSIC", "0"))
PARTS = os.environ.get("PARTS", "tree,pca,kmeans").split(",")
GRAPH_M = [int(x) for x in os.environ.get("GRAPH_M", "8,16,32").split(",")]
BASIS = (
    np.linalg.qr(np.random.default_rng(12345).standard_normal((D, INTRINSIC)))[0].T
    if INTRINSIC
    else None
)
META_WORDS = 4 + 65


def draw(g, n):
    if not INTRINSIC:
        return g.random((n, D))
    return g.random((n, INTRINSIC)) @ BASIS + 0.01 * g.standard_normal((n, D))


def build(rng):
    x0 = draw(rng, 1)
    m = EpistemicNearestNeighbors(x0, np.zeros((1, 1)), work_dir=WORK)
    xs, n = [x0], 1
    while n < N:
        b = min(1000, N - n)
        x = draw(rng, b)
        m.add(x, np.zeros((b, 1)))
        xs.append(x)
        n += b
    m.ensure_index_sync()
    return m, np.concatenate(xs, axis=0)


def tree_partition(work, n):
    metas = list(Path(work).rglob("tree_meta.bin"))
    w = np.fromfile(metas[0], dtype=np.uint32)
    w = w[: len(w) // META_WORDS * META_WORDS].reshape(-1, META_WORDS)
    kind, ln = w[:, 0] >> 24, w[:, 0] & 0xFFFFFF
    leaf = np.full(n, -1, dtype=np.int64)
    for j, pid in enumerate(np.flatnonzero((kind == 1) & (ln > 0))):
        ids = w[pid, 4 : 4 + ln[pid]].astype(np.int64)
        leaf[ids[ids < n]] = j
    return leaf


def split_mask(P, how, rng):
    if how == "pca":
        c = P - P.mean(0)
        v = np.linalg.svd(c[: min(len(c), 2000)], full_matrices=False)[2][0]
        s = c @ v
        return s <= np.median(s)
    sub = P[rng.choice(len(P), min(len(P), 4000), replace=False)]
    cen, _ = kmeans2(sub, 2, minit="++", seed=rng, iter=10)
    left = ((P - cen[0]) ** 2).sum(1) <= ((P - cen[1]) ** 2).sum(1)
    if left.all() or not left.any():
        left = np.arange(len(P)) < len(P) // 2
    return left


def split_rec(X, idx, cap, how, out, rng):
    stack = [idx]
    while stack:
        ix = stack.pop()
        if len(ix) <= cap:
            out.append(ix)
            continue
        left = split_mask(X[ix], how, rng)
        stack += [ix[left], ix[~left]]


def alt_partition(X, cap, how, seed):
    out = []
    split_rec(X, np.arange(len(X)), cap, how, out, np.random.default_rng(seed))
    leaf = np.empty(len(X), dtype=np.int64)
    for j, ix in enumerate(out):
        leaf[ix] = j
    return leaf


def lloyd_local(X, leaf, m, iters):
    leaf = leaf.copy()
    for _ in range(iters):
        _uniq, leaf = np.unique(leaf, return_inverse=True)
        cnt = np.bincount(leaf)
        cen = np.zeros((len(cnt), X.shape[1]))
        np.add.at(cen, leaf, X)
        cen /= cnt[:, None]
        nb = cKDTree(cen).query(cen, min(m + 1, len(cen)))[1]
        order = np.argsort(leaf, kind="stable")
        starts = np.r_[0, np.cumsum(cnt)]
        new = leaf.copy()
        for l in range(len(cnt)):
            ix = order[starts[l] : starts[l + 1]]
            c = nb[l]
            d = ((X[ix, None, :] - cen[c][None, :, :]) ** 2).sum(2)
            new[ix] = c[d.argmin(1)]
        leaf = new
    return np.unique(leaf, return_inverse=True)[1]


class Part:
    def __init__(self, X, leaf):
        self.order = np.argsort(leaf, kind="stable")
        sl = leaf[self.order]
        self.starts = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1]])
        self.leaf = leaf
        cnt = np.diff(np.r_[self.starts, len(sl)])
        Xs = X[self.order]
        self.cen = np.add.reduceat(Xs, self.starts, axis=0) / cnt[:, None]
        d = np.sqrt(((Xs - np.repeat(self.cen, cnt, axis=0)) ** 2).sum(1))
        self.rad = np.maximum.reduceat(d, self.starts)
        self.size = cnt
        self.kd = cKDTree(self.cen)
        self.graphs = {}

    def graph(self, m):
        if m not in self.graphs:
            self.graphs[m] = self.kd.query(self.cen, min(m + 1, len(self.cen)))[1][
                :, 1:
            ]
        return self.graphs[m]


def last_rank(order_keys, true_leaves):
    pos = np.empty(len(order_keys), dtype=np.int64)
    pos[np.argsort(order_keys, kind="stable")] = np.arange(len(order_keys))
    return int(pos[true_leaves].max()) + 1


def graph_rank(p, m, q, true_leaves, cap=20000):
    G = p.graph(m)
    need = set(true_leaves.tolist())
    start = int(p.kd.query(q, 1)[1])
    seen = {start}
    heap = [(float(((p.cen[start] - q) ** 2).sum()), start)]
    visits = 0
    while heap and visits < cap:
        _, u = heapq.heappop(heap)
        visits += 1
        need.discard(u)
        if not need:
            return visits
        nb = [int(v) for v in G[u] if v not in seen]
        if nb:
            dv = ((p.cen[nb] - q) ** 2).sum(1)
            for v, dd in zip(nb, dv):
                seen.add(v)
                heapq.heappush(heap, (float(dd), v))
    return -1


def query_stats(p, X, q, Xs):
    d2 = ((X - q) ** 2).sum(1)
    ex = np.argpartition(d2, K)[:K]
    kth = math.sqrt(float(d2[ex].max()))
    tl = np.unique(p.leaf[ex])
    mind = np.minimum.reduceat(d2[p.order], p.starts)
    dc = np.sqrt(((p.cen - q) ** 2).sum(1))
    rec = {
        "true_leaves": len(tl),
        "oracle": last_rank(mind, tl),
        "centroid": last_rank(dc, tl),
        "ball": last_rank(np.maximum(dc - p.rad, 0) + 1e-12 * dc, tl),
        "kth": kth,
        "touch": int((np.maximum(dc - p.rad, 0) <= kth).sum()),
        "rows_centroid": int(
            np.cumsum(p.size[np.argsort(dc, kind="stable")])[last_rank(dc, tl) - 1]
        ),
    }
    for m in GRAPH_M:
        rec[f"graph{m}"] = graph_rank(p, m, q, tl)
    return rec


def partitions(X, tree):
    cap = round(len(X) / (tree.max() + 1) * 4 / 3)
    parts = {"tree": tree}
    for how in ("pca", "kmeans"):
        if how in PARTS:
            parts[how] = alt_partition(X, max(cap, 16), how, SEED)
    for spec in [x for x in PARTS if x.startswith("lloyd")]:
        parts[spec] = lloyd_local(X, tree, 16, int(spec[5:] or 3))
    return {k: v for k, v in parts.items() if k in PARTS}


def record(name, p, X, leaf, recs, build_s):
    kth = np.median([r["kth"] for r in recs])
    return {
        "D": D,
        "n": N,
        "seed": SEED,
        "intrinsic": INTRINSIC,
        "part": name,
        "leaves": len(p.size),
        "rows_per_leaf": float(p.size.mean()),
        "size_p99": float(np.percentile(p.size, 99)),
        "rad_over_kth_p50": float(np.median(p.rad) / kth),
        "build_s": build_s,
        "env": {k: v for k, v in os.environ.items() if k.startswith("ENN_")},
        "misassigned": float(np.mean(p.kd.query(X, 1)[1] != leaf)),
        "queries": recs,
    }


def report(name, out, p, recs, build_s):
    keys = ["true_leaves", "oracle", "centroid", "rows_centroid", "ball", "touch"] + [
        f"graph{g}" for g in GRAPH_M
    ]
    s = " ".join(
        f"{k} {np.median([r[k] for r in recs]):.0f}/{np.percentile([r[k] for r in recs], 90):.0f}"
        for k in keys
    )
    print(
        f"D={D} N={N} I={INTRINSIC} {name:6s} build {build_s:.1f}s leaves {len(p.size)} rows/leaf {p.size.mean():.1f} "
        f"p99 {out['size_p99']:.0f} rad/kth {out['rad_over_kth_p50']:.2f} mis {out['misassigned']:.3f} | p50/p90 {s}",
        flush=True,
    )


def main():
    t0 = time.perf_counter()
    m, X = build(np.random.default_rng(SEED))
    build_s = time.perf_counter() - t0
    xq = draw(np.random.default_rng(10_000 + SEED), NQ)
    tree = tree_partition(WORK, len(X))
    assert (tree >= 0).all(), f"{(tree < 0).sum()} rows without a leaf"
    del m
    for name, leaf in partitions(X, tree).items():
        p = Part(X, leaf)
        recs = [query_stats(p, X, xq[i], None) for i in range(NQ)]
        out = record(name, p, X, leaf, recs, build_s)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(out) + "\n")
        report(name, out, p, recs, build_s)


if __name__ == "__main__":
    main()
