import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from partition_common import two_means
from scipy.spatial import cKDTree

import enn._rust as R
from enn.enn.enn_class import EpistemicNearestNeighbors

D, N, SEED = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
WORK, OUT = sys.argv[4], sys.argv[5]
ARM = os.environ.get("ARM", "r64")
CAP = int(os.environ.get("CAP", "64"))
NQ = int(os.environ.get("NQ", "300"))
K = int(os.environ.get("K", "10"))
INTRINSIC = int(os.environ.get("INTRINSIC", "0"))
OFFLINE = [int(c) for c in os.environ.get("OFFLINE", "").split(",") if c]
LLOYD = int(os.environ.get("LLOYD", "0"))
NOISE = 0.1
PAGE, PAGE_HDR, ROW_B = 4096, 64, 8 + D
GRID = np.unique(
    np.concatenate([np.arange(1, 10), np.geomspace(10, 20000, 60).astype(int)])
)
BASIS = (
    np.linalg.qr(np.random.default_rng(12345).standard_normal((D, INTRINSIC)))[0].T
    if INTRINSIC
    else None
)


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
    xs, ys, n = [x0.astype(np.float32)], [y0.astype(np.float32)], 1
    while n < N:
        b = min(1000, N - n)
        x = draw(rng, b)
        y = f(x) + NOISE * rng.standard_normal((b, 1))
        m.add(x, y)
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
        n += b
    m.ensure_index_sync()
    return m, np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)[:, 0]


def tree_partition(work, n, cap):
    words = 4 + cap + 1
    metas = list(Path(work).rglob("tree_meta.bin"))
    w = np.fromfile(metas[0], dtype=np.uint32)
    w = w[: len(w) // words * words].reshape(-1, words)
    kind, ln = w[:, 0] >> 24, w[:, 0] & 0xFFFFFF
    leaf = np.full(n, -1, dtype=np.int64)
    for j, pid in enumerate(np.flatnonzero((kind == 1) & (ln > 0))):
        ids = w[pid, 4 : 4 + ln[pid]].astype(np.int64)
        leaf[ids[ids < n]] = j
    return leaf


def offline_partition(X, cap, seed):
    rng = np.random.default_rng(seed)
    leaf = np.empty(len(X), dtype=np.int64)
    stack, j = [np.arange(len(X))], 0
    while stack:
        ix = stack.pop()
        if len(ix) <= cap:
            leaf[ix] = j
            j += 1
            continue
        left = two_means(X[ix], rng)
        stack += [ix[left], ix[~left]]
    return leaf


def lloyd_pass(X, leaf, m):
    cnt = np.bincount(leaf)
    cen = np.zeros((len(cnt), X.shape[1]), dtype=np.float64)
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
    return new


def lloyd_local(X, leaf, m, iters):
    for _ in range(iters):
        leaf = lloyd_pass(X, np.unique(leaf, return_inverse=True)[1], m)
    return np.unique(leaf, return_inverse=True)[1]


class Part:
    def __init__(self, X, leaf):
        self.leaf = leaf
        self.order = np.argsort(leaf, kind="stable")
        sl = leaf[self.order]
        self.starts = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1]])
        self.ends = np.r_[self.starts[1:], len(sl)]
        self.size = self.ends - self.starts
        self.cen = (
            np.add.reduceat(X[self.order].astype(np.float64), self.starts, axis=0)
            / self.size[:, None]
        )
        self.pages = np.ceil((PAGE_HDR + self.size * ROW_B) / PAGE).astype(np.int64)

    def rows(self, j):
        return self.order[self.starts[j] : self.ends[j]]

    def pages_of(self, rows):
        return self.pages[self.leaf[rows[0]]]

    def ideal_leaves(self, q, true_leaves):
        order = np.argsort(((self.cen - q) ** 2).sum(1), kind="stable")[:20000]
        pos = {int(j): t for t, j in enumerate(order)}
        last = max((pos.get(int(j), 19999) for j in true_leaves), default=0)
        return [self.rows(j) for j in order[: last + 1]]


class Query:
    def __init__(self, X, Y, q, fq):
        self.d2 = np.empty(len(X), dtype=np.float32)
        step = 1 << 20
        for s in range(0, len(X), step):
            blk = X[s : s + step] - q
            self.d2[s : s + step] = (blk * blk).sum(1)
        ex = np.argpartition(self.d2, K)[:K]
        self.ex = ex[np.argsort(self.d2[ex])]
        self.Y = Y
        self.fq = fq
        self.ex_err = float(Y[self.ex].mean() - fq)


def curve(by_leaf, pages, qry):
    top_d = np.empty(0)
    top_r = np.empty(0, dtype=np.int64)
    found, pg = 0, 0
    rec, err, pgs = [], [], []
    for rows in by_leaf:
        rows = np.asarray(rows, dtype=np.int64)
        v = qry.d2[rows]
        found += int(np.isin(rows, qry.ex).sum())
        pg += int(pages(rows))
        if len(top_d) < K or (v < top_d[-1]).any():
            td = np.concatenate([top_d, v])
            tr = np.concatenate([top_r, rows])
            o = np.argsort(td, kind="stable")[:K]
            top_d, top_r = td[o], tr[o]
        rec.append(found / K)
        err.append(float(qry.Y[top_r].mean() - qry.fq))
        pgs.append(pg)
    return np.asarray(rec), np.asarray(err), np.asarray(pgs)


def at_grid(a, fill):
    g = GRID[GRID <= len(a)]
    out = np.full(len(GRID), fill, dtype=np.float64)
    out[: len(g)] = a[g - 1]
    return out


def leaves_until_recall(rec):
    full = np.flatnonzero(rec >= 1.0)
    return int(full[0]) + 1 if len(full) else -1


def real_walk(m, q):
    m.neighbors(q[None, :].astype(np.float64), K)
    leaves, rows = R.take_walk_trace()
    nleaf = 1 + max((o for w, o, _ in rows if w == 1), default=-1)
    by_leaf = [[] for _ in range(nleaf)]
    for w, o, r in rows:
        if w == 1:
            by_leaf[o].append(r)
    walks = max((w for w, _, _ in leaves), default=1)
    return by_leaf, walks


def eval_order(by_leaf, pages_fn, qry, walks, tl):
    rec, err, pgs = curve(by_leaf, pages_fn, qry)
    ltr = leaves_until_recall(rec)
    return {
        "ltr": ltr,
        "pg_ltr": int(pgs[ltr - 1]) if ltr > 0 else -1,
        "L": len(by_leaf),
        "walks": walks,
        "tl": tl,
        "ex_err": qry.ex_err,
        "rec": at_grid(rec, rec[-1] if len(rec) else 0.0),
        "dmse": at_grid(err**2 - qry.ex_err**2, 0.0),
        "pgs": at_grid(pgs.astype(np.float64), pgs[-1] if len(pgs) else 0.0),
    }


def rank_stats(per):
    ltr = np.array([p["ltr"] for p in per])
    ok = ltr > 0
    pg = [p["pg_ltr"] for p in per if p["pg_ltr"] > 0]
    return {
        "n_q": len(per),
        "ltr_p50": float(np.median(ltr[ok])) if ok.any() else -1,
        "ltr_p90": float(np.percentile(ltr[ok], 90)) if ok.any() else -1,
        "ltr_mean": float(ltr[ok].mean()) if ok.any() else -1,
        "ltr_miss": float((~ok).mean()),
        "pages_at_ltr_p50": float(np.median(pg)) if pg else -1,
        "walk_leaves_mean": float(np.mean([p["L"] for p in per])),
        "walks_mean": float(np.mean([p["walks"] for p in per])),
        "true_leaves_mean": float(np.mean([p["tl"] for p in per])),
        "ex_mse": float(np.mean([p["ex_err"] ** 2 for p in per])),
    }


def curve_stats(per):
    n = len(per)
    dm = np.array([p["dmse"] for p in per])
    return {
        "grid": GRID.tolist(),
        "recall_grid": np.mean([p["rec"] for p in per], axis=0).tolist(),
        "dmse_grid": dm.mean(axis=0).tolist(),
        "dmse_se_grid": (dm.std(axis=0, ddof=1) / math.sqrt(n)).tolist(),
        "pages_grid": np.mean([p["pgs"] for p in per], axis=0).tolist(),
    }


def summarize(name, order, per):
    rec = {"arm": name, "order": order}
    rec.update(rank_stats(per))
    rec.update(curve_stats(per))
    return rec


def make_parts(X):
    parts, t_part = {ARM: Part(X, tree_partition(WORK, N, CAP))}, {}
    for c in OFFLINE:
        t1 = time.time()
        parts[f"o{c}"] = Part(X, offline_partition(X, c, SEED))
        t_part[f"o{c}"] = time.time() - t1
    if LLOYD and OFFLINE:
        t1 = time.time()
        k = f"l{OFFLINE[0]}"
        parts[k] = Part(X, lloyd_local(X, parts[f"o{OFFLINE[0]}"].leaf, 16, LLOYD))
        t_part[k] = time.time() - t1
    return parts, t_part


def eval_query(m, parts, qry, q, per):
    by_leaf, walks = real_walk(m, q)
    p_real = parts[ARM]
    tl_real = len(np.unique(p_real.leaf[qry.ex]))
    per[(ARM, "walk")].append(eval_order(by_leaf, p_real.pages_of, qry, walks, tl_real))
    for k, p in parts.items():
        tl_set = np.unique(p.leaf[qry.ex])
        leaves = p.ideal_leaves(q, tl_set)
        per[(k, "ideal")].append(eval_order(leaves, p.pages_of, qry, 1, len(tl_set)))
        if k == ARM:
            oracle = [p.rows(j) for j in tl_set]
            per[("oracle", "ideal")].append(
                eval_order(oracle, p.pages_of, qry, 1, len(tl_set))
            )


def run_queries(m, X, Y, parts):
    qrng = np.random.default_rng(SEED + 1000)
    xq = draw(qrng, NQ).astype(np.float32)
    fq = f(xq.astype(np.float64))[:, 0]
    per = {(ARM, "walk"): [], ("oracle", "ideal"): []}
    per.update({(k, "ideal"): [] for k in parts})
    R.set_walk_trace(True)
    for i in range(NQ):
        eval_query(m, parts, Query(X, Y, xq[i], fq[i]), xq[i], per)
    R.set_walk_trace(False)
    return per


def base_record(parts, t_build, t_part, t0):
    return {
        "D": D,
        "N": N,
        "seed": SEED,
        "intrinsic": INTRINSIC,
        "cap": CAP,
        "k": K,
        "nq": NQ,
        "lnN": math.log(N),
        "leaf_coef": float(R.leaf_coefficient()),
        "build_s": t_build,
        "part_s": t_part,
        "total_s": time.time() - t0,
        "env": {k: v for k, v in os.environ.items() if k.startswith("ENN_")},
        "n_leaves": {k: len(p.size) for k, p in parts.items()},
        "rows_per_leaf": {k: float(p.size.mean()) for k, p in parts.items()},
        "pages_per_leaf": {k: float(p.pages.mean()) for k, p in parts.items()},
    }


def write_out(base, per):
    with open(OUT, "a") as fh:
        for (name, order), lst in per.items():
            rec = dict(base)
            rec.update(summarize(name, order, lst))
            fh.write(json.dumps(rec) + "\n")
            print(
                f"{name:8s} {order:5s} ltr p50/p90 {rec['ltr_p50']:.0f}/{rec['ltr_p90']:.0f} "
                f"miss {rec['ltr_miss']:.3f} pages@ltr {rec['pages_at_ltr_p50']:.0f} "
                f"L {rec['walk_leaves_mean']:.0f} tl {rec['true_leaves_mean']:.1f}"
            )


def main():
    t0 = time.time()
    m, X, Y = build(np.random.default_rng(SEED))
    t_build = time.time() - t0
    parts, t_part = make_parts(X)
    per = run_queries(m, X, Y, parts)
    write_out(base_record(parts, t_build, t_part, t0), per)
    print(
        f"DONE D={D} N={N} seed={SEED} arm={ARM} build {t_build:.0f}s total {time.time() - t0:.0f}s"
    )


main()
