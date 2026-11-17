import json
import os
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

import enn._rust as r
from enn.enn.enn_class import EpistemicNearestNeighbors

KIND = {0: "leaf", 1: "upper_buf", 3: "leafparent_buf", 9: "absent"}


def draw(g, n, d):
    return g.random((n, d))


def f(x):
    return np.sin(3 * x[:, : min(5, x.shape[1])]).sum(1, keepdims=True)


def build(d, n, seed, work):
    rng = np.random.default_rng(seed)
    x0 = draw(rng, 1, d)
    m = EpistemicNearestNeighbors(x0, f(x0), work_dir=work)
    xs, done = [x0], 1
    while done < n:
        b = min(1000, n - done)
        x = draw(rng, b, d)
        m.add(x, f(x) + 0.1 * rng.standard_normal((b, 1)))
        xs.append(x)
        done += b
    m.ensure_index_sync()
    return m, np.concatenate(xs, axis=0)


def exact_knn(x, q, k):
    xx = (x * x).sum(1)
    out = np.empty((len(q), k), dtype=np.int64)
    for s in range(0, len(q), 20):
        b = q[s : s + 20]
        d2 = (b * b).sum(1)[:, None] + xx[None, :] - 2.0 * b @ x.T
        out[s : s + 20] = np.argpartition(d2, k, axis=1)[:, :k]
    return out


def tally(rows, found, ranks):
    cells = {}
    for row, kind, depth, _tag, scanned, popped, *rk in rows:
        key = (
            KIND.get(kind, str(kind)),
            int(depth),
            int(scanned),
            int(popped),
            int(row in found),
        )
        cells[key] = cells.get(key, 0) + 1
        if rk and kind in (0, 3):
            ranks.append([KIND[kind], int(row in found), int(rk[0]), int(rk[1])])
    return cells


def run_queries(m, q, exact, k):
    total, ranks, hits = {}, [], 0
    for i in range(len(q)):
        r.set_pbuf_diag_targets([int(v) for v in exact[i]])
        approx = set(m.neighbors(q[i : i + 1], k)[0].tolist())
        rows = r.take_pbuf_diag()
        r.set_pbuf_diag_targets([])
        hits += len(approx & set(exact[i].tolist()))
        for key, c in tally(rows, approx, ranks).items():
            total[key] = total.get(key, 0) + c
    return total, ranks, hits / (k * len(q))


def build_and_audit(d, n, seed, work):
    coef = float(os.environ.get("LEAF_COEF", "0") or 0)
    if coef > 0:
        r.set_leaf_coefficient(coef)
    m, x = build(d, n, seed, work)
    moved = r.take_pbuf_rehome_stats()
    audit = [[int(a[0]), int(a[6]), int(a[7])] for a in r.take_pbuf_diag() if a[1] == 7]
    return m, x, moved, audit


def main():
    d, n, seed = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
    work, out = sys.argv[4], sys.argv[5]
    k, nq = 10, int(os.environ.get("NQ", "300"))
    m, x, moved, audit = build_and_audit(d, n, seed, work)
    q = draw(np.random.default_rng(10_000 + seed), nq, d)
    total, ranks, recall = run_queries(m, q, exact_knn(x, q, k), k)
    rec = {
        "arm": os.environ.get("ARM", ""),
        "D": d,
        "n": n,
        "seed": seed,
        "recall": recall,
        "rehome_checked": moved[0],
        "rehome_moved": moved[1],
        "cells": [list(key) + [c] for key, c in sorted(total.items())],
        "ranks": ranks,
        "audit": audit,
    }
    print(json.dumps(rec), flush=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
