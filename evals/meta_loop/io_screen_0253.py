import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

import enn._rust as R
from enn.enn.enn_class import EpistemicNearestNeighbors

D, N, SEED, WORK, OUT = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    int(sys.argv[3]),
    sys.argv[4],
    sys.argv[5],
)
NQ = int(os.environ.get("NQ", "200"))
INTRINSIC = int(os.environ.get("INTRINSIC", "0"))
GAP = int(os.environ.get("GAP", "2"))
META_WORDS = 4 + 65
SIB = 16
K = 10
BASIS = np.random.default_rng(99).standard_normal((max(INTRINSIC, 1), D))


def draw(g, n):
    if not INTRINSIC:
        return g.random((n, D))
    return g.random((n, INTRINSIC)) @ BASIS + 0.01 * g.standard_normal((n, D))


def build(rng):
    x0 = draw(rng, 1)
    m = EpistemicNearestNeighbors(x0, np.sin(x0.sum(1, keepdims=True)), work_dir=WORK)
    n = 1
    while n < N:
        b = min(1000, N - n)
        x = draw(rng, b)
        m.add(x, np.sin(x.sum(1, keepdims=True)))
        n += b
    m.ensure_index_sync()
    return m


def load_meta():
    meta = next(Path(WORK).rglob("tree_meta.bin"))
    w = np.fromfile(meta, dtype=np.uint32)
    return w[: len(w) // META_WORDS * META_WORDS].reshape(-1, META_WORDS)


def maps(w):
    kind, ln = w[:, 0] >> 24, w[:, 0] & 0xFFFFFF
    row_leaf = np.full(N + 1, -1, dtype=np.int64)
    leaf_parent = np.full(len(w), -1, dtype=np.int64)
    for pid in np.flatnonzero((kind == 1) & (ln > 0)):
        ids = w[pid, 4 : 4 + ln[pid]].astype(np.int64)
        row_leaf[ids[ids <= N]] = pid
    for pid in np.flatnonzero(kind == 2):
        kids = w[pid, 4 : 4 + ln[pid]].astype(np.int64)
        kids = kids[kids < len(w)]
        if len(kids) and np.all(kind[kids] == 1):
            leaf_parent[kids] = pid
    is_leaf = kind == 1
    return row_leaf, leaf_parent, is_leaf


def window_runs(seq, win):
    total = 0
    for i in range(0, len(seq), win):
        p = np.unique(seq[i : i + win])
        total += 1 + int(np.sum(np.diff(p) > GAP))
    return total


def per_query(rows, row_leaf, leaf_parent, is_leaf):
    by = {}
    for w, o, r in rows:
        by.setdefault((w, o), r)
    seq = [row_leaf[by[key]] for key in sorted(by)]
    seq = np.array([p for p in seq if p >= 0], dtype=np.int64)
    first = np.unique(seq, return_index=True)[1]
    seq = seq[np.sort(first)]
    pages = np.unique(seq)
    if len(pages) == 0:
        return None
    blocks = np.unique(pages // SIB)
    runs = 1 + int(np.sum(np.diff(pages) > GAP))
    alloc = sum(int(is_leaf[b * SIB : (b + 1) * SIB].sum()) for b in blocks)
    return {
        "leaves": len(pages),
        "blocks": len(blocks),
        "runs": runs,
        "parents": len(np.unique(leaf_parent[pages])),
        "block_util": len(pages) / max(1, alloc),
        "la10_runs": window_runs(seq, 10),
        "la30_runs": window_runs(seq, 30),
    }


def trace_queries(m, row_leaf, leaf_parent, is_leaf):
    q = draw(np.random.default_rng(10_000 + SEED), NQ)
    R.set_walk_trace(True)
    recs = []
    for i in range(NQ):
        m.neighbors(q[i : i + 1], K)
        _, rows = R.take_walk_trace()
        r = per_query(rows, row_leaf, leaf_parent, is_leaf)
        if r:
            recs.append(r)
    R.set_walk_trace(False)
    return recs


def main():
    m = build(np.random.default_rng(SEED))
    w = load_meta()
    row_leaf, leaf_parent, is_leaf = maps(w)
    recs = trace_queries(m, row_leaf, leaf_parent, is_leaf)
    summ = {
        key: float(np.mean([r[key] for r in recs]))
        for key in (
            "leaves",
            "blocks",
            "runs",
            "parents",
            "block_util",
            "la10_runs",
            "la30_runs",
        )
    }
    out = {
        "D": D,
        "N": N,
        "seed": SEED,
        "intrinsic": INTRINSIC,
        "extent": os.environ.get("ENN_LEAF_EXTENT", "0"),
        "budget": math.ceil(R.leaf_coefficient() * math.log(N)),
        "leaf_pages_total": int(is_leaf.sum()),
        "pages_total": len(w),
        "code_page_bytes": next(Path(WORK).rglob("tree_codes.bin")).stat().st_size
        // len(w),
        **summ,
    }
    print(json.dumps(out), flush=True)
    Path(OUT).write_text(json.dumps({"summary": out, "per_query": recs}))


main()
