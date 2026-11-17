import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from scipy.spatial import cKDTree

from enn.enn.enn_class import EpistemicNearestNeighbors

SLOTS = 65
META = 4 + SLOTS
WIDE = int(os.environ.get("WIDE", "16"))
NO_PARENT = 0xFFFFFFFF


def f(x):
    return np.sin(3 * x[:, : min(5, x.shape[1])]).sum(1, keepdims=True)


def build(d, n, seed, work):
    rng = np.random.default_rng(seed)
    x0 = rng.random((1, d))
    m = EpistemicNearestNeighbors(x0, f(x0), work_dir=work)
    done = 1
    while done < n:
        b = min(1000, n - done)
        x = rng.random((b, d))
        m.add(x, f(x) + 0.1 * rng.standard_normal((b, 1)))
        done += b
    m.ensure_index_sync()
    return m


class Tree:
    def __init__(self, work, d):
        meta = np.fromfile(next(Path(work).rglob("tree_meta.bin")), dtype=np.uint32)
        self.m = meta[: len(meta) // META * META].reshape(-1, META)
        sl = np.fromfile(next(Path(work).rglob("tree_slots.bin")), dtype=np.float32)
        self.s = sl[: len(sl) // (SLOTS * d) * (SLOTS * d)].reshape(-1, SLOTS * d)
        self.d = d
        kind, ln = self.m[:, 0] >> 24, self.m[:, 0] & 0xFFFFFF
        roots = np.flatnonzero((kind == 2) & (self.m[:, 3] == NO_PARENT) & (ln > 0))
        self.root = int(roots[0])
        self.kids, self.cent, self.cnt, self.level = {}, {}, {}, {}
        stack = [(self.root, 0)]
        while stack:
            p, lv = stack.pop()
            self.level[p] = lv
            self.cnt[p] = int(self.m[p, 1])
            if kind[p] != 2:
                continue
            ids = [int(c) for c in self.m[p, 4 : 4 + ln[p]]]
            blk = self.s[min(p, len(self.s) - 1), : len(ids) * d].reshape(-1, d)
            self.kids[p] = ids
            for c, v in zip(ids, blk):
                self.cent[c] = v.astype(np.float64)
                stack.append((c, lv + 1))
        self.leaves = [p for p in self.level if p not in self.kids]
        self.cent[self.root] = np.zeros(d)

    def parent_of(self):
        return {c: p for p, ks in self.kids.items() for c in ks}

    def descend(self, q):
        p = self.root
        while p in self.kids:
            ks = self.kids[p]
            c = np.stack([self.cent[k] for k in ks])
            p = ks[int(((c - q) ** 2).sum(1).argmin())]
        return p

    def owner_ranks(self, P):
        L = np.array(self.leaves)
        C = np.stack([self.cent[k] for k in L])
        pos = {int(k): i for i, k in enumerate(L)}
        out = np.empty(len(P), dtype=np.int64)
        for i, q in enumerate(P):
            d2 = ((C - q) ** 2).sum(1)
            out[i] = int((d2 < d2[pos[self.descend(q)]]).sum())
        return out

    def parent_ranks(self):
        par = self.parent_of()
        lp = sorted({par[k] for k in self.leaves})
        C = np.stack([self.cent[p] for p in lp])
        pos = {p: i for i, p in enumerate(lp)}
        r = []
        for k in self.leaves:
            d2 = ((C - self.cent[k]) ** 2).sum(1)
            r.append(int((d2 < d2[pos[par[k]]]).sum()))
        return np.array(r)

    def depth(self):
        return max(self.level.values())


def weighted_mean(t, ks):
    w = np.array([max(t.cnt[k], 1) for k in ks], dtype=np.float64)
    return (np.stack([t.cent[k] for k in ks]) * w[:, None]).sum(0) / w.sum()


def nodes_at(t, lv):
    return [p for p, l in t.level.items() if l == lv]


def candidates(t, kids, par, mode, hood):
    ps = sorted({par[k] for k in kids})
    if mode == "all":
        return {k: ps for k in kids}
    C = np.stack([t.cent[k] for k in kids])
    nb = cKDTree(C).query(C, min(hood + 1, len(kids)))[1]
    return {
        k: sorted({par[kids[j]] for j in np.atleast_1d(nb[i])})
        for i, k in enumerate(kids)
    }


def move_one(t, k, ps, par):
    P = np.stack([t.cent[p] for p in ps])
    for j in np.argsort(((P - t.cent[k]) ** 2).sum(1)):
        dst, src = ps[j], par[k]
        if dst == src:
            return 0
        if len(t.kids[dst]) >= WIDE or len(t.kids[src]) <= 1:
            continue
        t.kids[src].remove(k)
        t.kids[dst].append(k)
        par[k] = dst
        return 1
    return 0


def refit_level(t, kids, par):
    for p in {par[k] for k in kids}:
        t.cent[p] = weighted_mean(t, t.kids[p])
        t.cnt[p] = sum(t.cnt[k] for k in t.kids[p])


def reparent_level(t, lv, mode, hood, iters, rng):
    par = t.parent_of()
    kids = nodes_at(t, lv + 1)
    if len({par[k] for k in kids}) < 2:
        return 0
    moved = 0
    for _ in range(iters):
        if mode != "none":
            cand = candidates(t, kids, par, mode, hood)
            moved += sum(
                move_one(t, int(k), cand[int(k)], par) for k in rng.permutation(kids)
            )
        refit_level(t, kids, par)
    return moved


def reparent(t, levels, mode, hood, iters, seed):
    rng = np.random.default_rng(seed)
    deep = t.depth()
    moved = []
    for lv in range(deep - 1, deep - 1 - levels, -1):
        if lv < 1:
            break
        moved.append(reparent_level(t, lv, mode, hood, iters, rng))
    return moved


def summary(t, P):
    o = t.owner_ranks(P)
    pr = t.parent_ranks()
    return {
        "owner_med": float(np.median(o)),
        "owner_p90": float(np.percentile(o, 90)),
        "owner_hit": float((o == 0).mean()),
        "parent_med": float(np.median(pr)),
        "parent_hit": float((pr == 0).mean()),
    }


def header(m, work, d, n, seed, t0):
    import enn._rust as R

    base = Tree(work, d)
    rec = {
        "D": d,
        "n": n,
        "seed": seed,
        "depth": base.depth(),
        "leaves": len(base.leaves),
    }
    rec["build_s"] = time.time() - t0
    rec["reparent"] = list(getattr(R, "take_reparent_stats", lambda: (0, 0))())
    return base, rec


def run_arms(rec, work, d, seed, P):
    arms = os.environ.get(
        "ARMS", "1:graph:16:1 1:graph:16:3 9:graph:16:3 9:all:0:3"
    ).split()
    for a in arms:
        lv, mode, hood, iters = a.split(":")
        t = Tree(work, d)
        moved = reparent(t, int(lv), mode, int(hood), int(iters), seed)
        rec[a] = summary(t, P) | {"moved": moved}
        print(a, rec[a], flush=True)


def main():
    d, n, seed = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
    work, out = sys.argv[4], sys.argv[5]
    t0 = time.time()
    m = build(d, n, seed, work)
    P = np.random.default_rng(20_000 + seed).random(
        (int(os.environ.get("NP", "2000")), d)
    )
    base, rec = header(m, work, d, n, seed, t0)
    rec["base"] = summary(base, P)
    run_arms(rec, work, d, seed, P)
    del m
    print(json.dumps(rec), flush=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
