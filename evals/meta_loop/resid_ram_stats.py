import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def main():
    rows = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines()]
    by = {(r["advise"], r["seed"]): r for r in rows}
    BASE = "c4_base" if any(r["advise"] == "c4_base" for r in rows) else "c4_0"
    seeds = sorted({r["seed"] for r in rows})
    for k in ("add_us_row", "q_us_med"):
        base = np.array([by[(BASE, s)][k] for s in seeds])
        print(k, BASE, "mean", round(base.mean(), 3))
        for a in sorted({r["advise"] for r in rows} - {BASE}):
            v = np.array([by[(a, s)][k] for s in seeds])
            t = stats.ttest_rel(v, base)
            print(
                f"  {a}: mean {v.mean():.3f} ratio {np.mean(v / base):.3f} SE(diff) "
                f"{np.std(v - base, ddof=1) / np.sqrt(len(v)):.3f} p {t.pvalue:.2g}"
            )


if __name__ == "__main__":
    main()
