import ast
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def rd(p):
    return [ast.literal_eval(l) for l in Path(p).read_text().splitlines()]


def diff_stats(pr, k):
    x = np.array([b[k] - m[k] for b, m in pr])
    se = x.std(ddof=1) / np.sqrt(len(x))
    if np.allclose(x, 0):
        return [0.0, 0.0, 0.0, 1.0]
    t, p = stats.ttest_1samp(x, 0.0)
    return [x.mean(), se, t, p]


def row(B, md, D, n):
    pr = [
        (b, md[(D, n, b["seed"])])
        for b in B
        if b["D"] == D and b["n"] == n and (D, n, b["seed"]) in md
    ]
    out = [D, n, len(pr)]
    for k in ("loglik", "nrmse"):
        out += diff_stats(pr, k)
    out += [
        np.mean([b["add_us_row"] / m["add_us_row"] for b, m in pr]),
        np.mean([b["query_s"] / m["query_s"] for b, m in pr]),
    ]
    return ",".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in out)


def main():
    B, M = rd(sys.argv[1]), rd(sys.argv[2])
    md = {(d["D"], d["n"], d["seed"]): d for d in M}
    print("D,N,n_seeds,dloglik,se,t,p,dnrmse,se,t,p,add_ratio,query_ratio")
    for D in (10, 100):
        for n in (1000, 10000, 100000):
            print(row(B, md, D, n))


if __name__ == "__main__":
    main()
