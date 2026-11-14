import ast
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def main():

    rd = lambda p: [ast.literal_eval(l) for l in Path(p).read_text().splitlines()]
    M = rd(sys.argv[1])
    md = {(d["D"], d["n"], d["seed"]): d for d in M}
    rec = {}
    for f in sys.argv[2:]:
        for r in rd(f):
            if r["n"] == 100000:
                rec[(r["D"], r["leaf_coef"], r["seed"])] = r
    rows = {}
    for (D, L, s), r in rec.items():
        rows.setdefault((D, L), []).append(r)
    print(
        "D,L,n,recall,dloglik,se,p,dnrmse,se,p,query_ratio_vs_main,knn_gap_approx_minus_exact"
    )
    for (D, L), rs in sorted(rows.items()):
        pr = [(r, md[(D, 100000, r["seed"])]) for r in rs]
        o = [D, L, len(pr), np.mean([r["recall_at_10"] for r, _ in pr])]
        for k in ("loglik", "nrmse"):
            x = np.array([r[k] - m[k] for r, m in pr])
            o += [
                x.mean(),
                x.std(ddof=1) / np.sqrt(len(x)),
                stats.ttest_1samp(x, 0).pvalue,
            ]
        o.append(np.mean([r["query_s"] / m["query_s"] for r, m in pr]))
        o.append(np.mean([r["knn_nrmse_approx"] - r["knn_nrmse_exact"] for r, _ in pr]))
        print(",".join(f"{v:.4g}" for v in o))


if __name__ == "__main__":
    main()
