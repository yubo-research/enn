import json
import sys

import numpy as np
from scipy import stats


def load(p):
    return {(o["D"], o["n"], o["seed"]): o for o in map(json.loads, open(p))}


a, b = load(sys.argv[1]), load(sys.argv[2])
for D, n in sorted({(k[0], k[1]) for k in a if k in b}):
    ks = [k for k in a if k in b and k[:2] == (D, n)]
    out = [f"D={D} n={n:.0e} seeds={len(ks)}"]
    for m in ("loglik", "nrmse", "recall_at_10", "neighbors_ms_per_query"):
        d = np.array([a[k][m] - b[k][m] for k in ks])
        p = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 else float("nan")
        out.append(
            f"d{m} {d.mean():+.4g} (SE {d.std(ddof=1) / np.sqrt(len(d)):.2g}, p={p:.2g}; B {np.mean([b[k][m] for k in ks]):.4g})"
        )
    print("; ".join(out))
