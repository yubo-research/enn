import json
import math
import os
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

import enn._rust as R
from enn.enn.enn_class import EpistemicNearestNeighbors

D, N, SEED = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
WORK, OUT = sys.argv[4], sys.argv[5]
NQ = int(os.environ.get("NQ", "200"))
K = int(os.environ.get("K", "10"))
INTRINSIC = int(os.environ.get("INTRINSIC", "0"))
BASIS = (
    np.linalg.qr(np.random.default_rng(12345).standard_normal((D, INTRINSIC)))[0].T
    if INTRINSIC
    else None
)
NOISE = 0.1
FIXED_M = (3, 10, 30, 100, 300)
PER_LN = (1, 3, 10, 30)


def draw(g, n):
    if not INTRINSIC:
        return g.random((n, D))
    return g.random((n, INTRINSIC)) @ BASIS + 0.01 * g.standard_normal((n, D))


def f(x):
    z = x @ BASIS.T if INTRINSIC else x
    return np.sin(3 * z[:, : min(5, z.shape[1])]).sum(1, keepdims=True)


def build(rng):
    x0 = draw(rng, 1)
    y0 = f(x0) + NOISE * rng.standard_normal((1, 1))
    m = EpistemicNearestNeighbors(x0, y0, work_dir=WORK)
    xs, ys, n = [x0], [y0], 1
    while n < N:
        b = min(1000, N - n)
        x = draw(rng, b)
        y = f(x) + NOISE * rng.standard_normal((b, 1))
        m.add(x, y)
        xs.append(x)
        ys.append(y)
        n += b
    m.ensure_index_sync()
    return m, np.concatenate(xs, axis=0), np.concatenate(ys, axis=0).ravel()


def prefix_curves(by_leaf, d2, ranks):
    n = len(by_leaf)
    changed = np.zeros(n, dtype=bool)
    full = np.zeros(n, dtype=bool)
    top = np.empty(0)
    for i, rows in enumerate(by_leaf):
        v = d2[np.asarray(rows, dtype=np.int64)]
        kth = top[-1] if len(top) >= K else np.inf
        changed[i] = bool((v < kth).any())
        if changed[i]:
            top = np.sort(np.concatenate([top, v]))[:K]
        full[i] = len(top) >= K
    r = np.asarray([x for x in ranks if x >= 0], dtype=np.int64)
    recall = np.bincount(r, minlength=n)[:n].cumsum() / K if n else np.zeros(0)
    return changed, full, recall


def leaf_mindists(by_leaf, X, q, d2):
    md = np.empty(len(by_leaf))
    mind = np.empty(len(by_leaf))
    for i, rows in enumerate(by_leaf):
        idx = np.asarray(rows, dtype=np.int64)
        P = X[idx]
        g = np.maximum(P.min(0) - q, 0) + np.maximum(q - P.max(0), 0)
        md[i] = float((g * g).sum())
        mind[i] = float(d2[idx].min())
    return md, mind


def rank_positions(keys):
    order = np.argsort(keys, kind="stable")
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    return pos


def true_leaves_of(by_leaf, exact):
    leaf_of = {}
    for i, rows in enumerate(by_leaf):
        for r in rows:
            leaf_of.setdefault(r, i)
    return [leaf_of.get(int(e), -1) for e in exact]


def box_stats(by_leaf, X, q, d2, exact, kth):
    if not by_leaf:
        return {}
    md, mind = leaf_mindists(by_leaf, X, q, d2)
    pos, opos = rank_positions(md), rank_positions(mind)
    tl = true_leaves_of(by_leaf, exact)
    ok = all(t >= 0 for t in tl)
    return {
        "box_open": int((md < kth).sum()),
        "box_last_true_rank": int(max(pos[t] for t in tl)) if ok else -1,
        "oracle_last_true_rank": int(max(opos[t] for t in tl)) if ok else -1,
        "leaf_touch_kth": int((mind <= kth).sum()),
    }


def replay(curves, m):
    changed, full, recall = curves
    n = len(changed)
    if n == 0:
        return 0, 0.0
    quiet = 0
    for i in range(n):
        quiet = 0 if changed[i] else quiet + 1
        if m is not None and full[i] and quiet >= m:
            return i + 1, float(recall[i])
    return n, float(recall[-1])


def quality(stop, by_leaf, d2, Y, ex, fq, ex_err):
    leaves_used, rec_ = stop
    pref = np.fromiter((r for b in by_leaf[:leaves_used] for r in b), dtype=np.int64)
    top = pref[np.argpartition(d2[pref], K - 1)[:K]] if len(pref) > K else pref
    top = top[np.argsort(d2[top])]
    ratio = float(np.mean(np.sqrt(d2[top]) / np.sqrt(np.maximum(d2[ex], 1e-300))))
    return [leaves_used, rec_, ratio, float(Y[top].mean() - fq), ex_err]


def first_walk_leaves(rows):
    w1 = [r for r in rows if r[0] == 1]
    nleaf = 1 + max((o for _, o, _ in w1), default=-1)
    by_leaf = [[] for _ in range(nleaf)]
    first = {}
    for _, o, r in w1:
        by_leaf[o].append(r)
        first.setdefault(r, o)
    return w1, by_leaf, first


def walk_record(approx, leaves, w1, by_leaf, ranks, exact, kth):
    nleaf = len(by_leaf)
    return {
        "walks": 1 + max((w for w, _, _ in leaves), default=0) - 1,
        "leaves": nleaf,
        "rows": len(w1),
        "recall_api": len(set(map(int, approx)) & set(exact.tolist())) / K,
        "last_true_rank": max(ranks) if min(ranks) >= 0 else -1,
        "missing_true": sum(r < 0 for r in ranks),
        "bound_gt_kth": int(sum(b > kth for w, _, b in leaves if w == 1)),
        "rows_per_leaf": len(w1) / max(1, nleaf),
        "true_leaves": len({x for x in ranks if x >= 0}),
    }


def query_record(m, X, Y, q, lnn):
    approx = m.neighbors(q, K)[0]
    leaves, rows = R.take_walk_trace()
    d2 = ((X - q) ** 2).sum(1)
    exact = np.argpartition(d2, K)[:K]
    w1, by_leaf, first = first_walk_leaves(rows)
    ranks = [first.get(int(e), -1) for e in exact]
    kth = float(np.sort(d2[exact])[-1])
    rec = walk_record(approx, leaves, w1, by_leaf, ranks, exact, kth)
    cur = prefix_curves(by_leaf, d2, ranks)
    rec.update(box_stats(by_leaf, X, q[0], d2, exact, kth))
    fq = float(f(q)[0, 0])
    ex = exact[np.argsort(d2[exact])]
    ex_err = float(Y[ex].mean() - fq)
    stops = {"full": None}
    stops.update({f"m{mm}": mm for mm in FIXED_M})
    stops.update({f"p{p}": max(1, round(p * lnn)) for p in PER_LN})
    for key, lim in stops.items():
        rec[key] = quality(replay(cur, lim), by_leaf, d2, Y, ex, fq, ex_err)
    return rec


def print_headline(per_q):
    L = np.array([r["leaves"] for r in per_q])
    ltr = np.array([r["last_true_rank"] for r in per_q])
    print(
        f"D={D} N={N} I={INTRINSIC} leaves/q {L.mean():.0f} recall_api "
        f"{np.mean([r['recall_api'] for r in per_q]):.4f} last_true_rank p50/p90/p99/max "
        f"{np.percentile(ltr, [50, 90, 99]).round(0).tolist()} {ltr.max()} miss {np.mean(ltr < 0):.3f}"
    )
    print(
        f"  rows/leaf {np.mean([r['rows_per_leaf'] for r in per_q]):.1f} walks/q "
        f"{np.mean([r['walks'] for r in per_q]):.2f} leaves with bound>kth "
        f"{np.mean([r['bound_gt_kth'] for r in per_q]):.1f}"
    )


def print_summary(per_q):
    print_headline(per_q)
    for key in (
        "true_leaves",
        "leaf_touch_kth",
        "box_open",
        "box_last_true_rank",
        "oracle_last_true_rank",
    ):
        a = np.array([r.get(key, -1) for r in per_q])
        print(
            f"  {key:22s} mean {a.mean():7.1f} p50 {np.median(a):6.0f} p90 {np.percentile(a, 90):6.0f} p99 {np.percentile(a, 99):6.0f}"
        )
    for key in ["full"] + [f"m{x}" for x in FIXED_M] + [f"p{x}" for x in PER_LN]:
        a = np.array([r[key] for r in per_q])
        d = a[:, 3] ** 2 - a[:, 4] ** 2
        print(
            f"  {key:5s} leaves {a[:, 0].mean():7.1f} recall {a[:, 1].mean():.4f} dist_ratio "
            f"{a[:, 2].mean():.4f} knn_rmse {np.sqrt(np.mean(a[:, 3] ** 2)):.4f} "
            f"(exact {np.sqrt(np.mean(a[:, 4] ** 2)):.4f}) paired_dmse "
            f"{np.mean(d):+.5f} se {np.std(d) / np.sqrt(len(a)):.5f}"
        )


def main():
    m, X, Y = build(np.random.default_rng(SEED))
    xq = draw(np.random.default_rng(10_000 + SEED), NQ)
    lnn = math.log(N)
    if os.environ.get("RADIUS_W"):
        R.set_radius_weight(float(os.environ["RADIUS_W"]))
    R.set_walk_trace(True)
    R.take_walk_totals()
    per_q = [query_record(m, X, Y, xq[i : i + 1], lnn) for i in range(NQ)]
    R.set_walk_trace(False)
    out = {
        "D": D,
        "n": N,
        "seed": SEED,
        "intrinsic": INTRINSIC,
        "radius_w": os.environ.get("RADIUS_W", "default"),
        "k": K,
        "budget": math.ceil(R.leaf_coefficient() * lnn),
        "leaf_coef": float(R.leaf_coefficient()),
        "lnN": lnn,
        "queries": per_q,
    }
    with open(OUT, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print_summary(per_q)


if __name__ == "__main__":
    main()
