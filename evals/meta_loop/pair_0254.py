import json
import sys

import numpy as np
from scipy import stats

KEYS = (
    "loglik",
    "nrmse",
    "recall_at_10",
    "add_us_row",
    "query_s",
    "neighbors_ms_per_query",
)


def load(p):
    return {(o["D"], o["n"], o["seed"]): o for o in map(json.loads, open(p))}


a, b = load(sys.argv[1]), load(sys.argv[2])
want = (int(sys.argv[3]), int(float(sys.argv[4]))) if len(sys.argv) > 4 else None
for D, n in sorted({k[:2] for k in a if k in b}):
    if want and (D, n) != want:
        continue
    ks = [k for k in a if k in b and k[:2] == (D, n)]
    out = [f"D={D} n={n:.0e} seeds={len(ks)}"]
    for m in KEYS:
        if not all(m in a[k] and m in b[k] for k in ks):
            continue
        x, y = np.array([a[k][m] for k in ks]), np.array([b[k][m] for k in ks])
        p = stats.ttest_rel(x, y).pvalue if len(ks) > 1 else float("nan")
        se = np.std(x - y, ddof=1) / np.sqrt(len(ks))
        out.append(
            f"{m}: A {x.mean():.5g} B {y.mean():.5g} d {np.mean(x - y):+.4g} (SE {se:.2g}, p={p:.2g}, ratio {x.mean() / y.mean():.3f})"
        )
    print("\n  ".join(out))
