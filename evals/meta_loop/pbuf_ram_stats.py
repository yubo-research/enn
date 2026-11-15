import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def main():
    rows = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines()]
    base_label = sys.argv[2]
    by = {(r["advise"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows if (base_label, r["seed"]) in by})
    for k in ("add_us_row", "q_us_med"):
        base = np.array([by[(base_label, s)][k] for s in seeds])
        print(k, base_label, "mean", round(base.mean(), 3), "n", len(seeds))
        for a in sorted({r["advise"] for r in rows} - {base_label}):
            ss = [s for s in seeds if (a, s) in by]
            v = np.array([by[(a, s)][k] for s in ss])
            b = np.array([by[(base_label, s)][k] for s in ss])
            t = stats.ttest_rel(v, b)
            print(
                f"  {a}: mean {v.mean():.3f} ratio {np.mean(v / b):.3f} SE(diff) "
                f"{np.std(v - b, ddof=1) / np.sqrt(len(v)):.3f} p {t.pvalue:.2g}"
            )


if __name__ == "__main__":
    main()
