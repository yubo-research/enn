import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

rows = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines()]
d = collections.defaultdict(dict)
for r in rows:
    d[(r["prob"], r["n"], r["seed"])][r["mode"]] = r
for pn in ("sparse", "ranges", "shifted"):
    for n in (100, 1000, 10000, 30000):
        for m in ("loglik", "nrmse"):
            diff = np.array(
                [
                    v["f32"][m] - v["f64"][m]
                    for k, v in d.items()
                    if k[0] == pn and k[1] == n
                ]
            )
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            t = stats.ttest_1samp(diff, 0)
            w = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            print(
                f"{pn} {n} {m} n={len(diff)} mean_diff={diff.mean():+.2e} se={se:.2e} p_t={t.pvalue:.3f} p_w={w:.3f} maxabs={abs(diff).max():.2e}"
            )
