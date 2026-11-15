import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load(d, D, cap, a):
    out = {}
    for p in sorted(Path(d).glob(f"c4_D{D}_{cap}_a{a}_r*.jsonl")):
        r = p.stem.split("_r")[-1]
        for o in map(json.loads, p.read_text().splitlines()):
            out[(r, o["n"])] = o
    return out


def med(O, keys, f):
    return float(np.median([np.median(O[k][f]) for k in keys]))


def add(O, keys):
    return float(np.median([O[k]["add_us_row"] or np.nan for k in keys]))


def row(a, n, A, B):
    keys = [k for k in A if k[1] == n and k in B]
    ca = np.concatenate([np.array(A[k]["cold"]) for k in keys]) / 1e3
    cb = np.concatenate([np.array(B[k]["cold"]) for k in keys]) / 1e3
    lr = np.log(ca / cb)
    p = stats.wilcoxon(lr).pvalue if n > 1 and np.any(lr != 0) else 1.0
    g = float(np.exp(lr.mean()))
    return {
        "arm": a,
        "n": n,
        "reps": len(keys),
        "cold_ms": float(np.median(ca)),
        "base_cold_ms": float(np.median(cb)),
        "ratio": g,
        "ratio_se": g * float(lr.std(ddof=1)) / np.sqrt(len(lr)),
        "p": float(p),
        "pages": med(A, keys, "rd") / 4096,
        "base_pages": med(B, keys, "rd") / 4096,
        "warm_us": med(A, keys, "warm"),
        "base_warm_us": med(B, keys, "warm"),
        "add_us": add(A, keys),
        "base_add_us": add(B, keys),
    }


def main():
    d, D, cap, base, arms = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5:],
    )
    B = load(d, D, cap, base)
    for a in arms:
        A = load(d, D, cap, a)
        for n in sorted({k[1] for k in A} & {k[1] for k in B}):
            print(json.dumps(row(a, n, A, B)))


if __name__ == "__main__":
    main()
