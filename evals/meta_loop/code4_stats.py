import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load(d, D, cap, a, r):
    p = Path(d) / f"c4_D{D}_{cap}_a{a}_r{r}.jsonl"
    return (
        {o["n"]: o for o in map(json.loads, p.read_text().splitlines())}
        if p.exists()
        else {}
    )


def main():
    d, D, cap, reps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    print(
        "rep,n,cold_off_ms,cold_on_ms,ratio,p,pages_off,pages_on,warm_off_us,warm_on_us,add_off,add_on,mu_absdiff_med"
    )
    for r in reps:
        A, B = load(d, D, cap, 0, r), load(d, D, cap, 1, r)
        for n in sorted(set(A) & set(B)):
            a, b = A[n], B[n]
            ca, cb = np.array(a["cold"]) / 1e3, np.array(b["cold"]) / 1e3
            p = stats.wilcoxon(ca, cb).pvalue if n > 1 else 1.0
            f = lambda o, k: float(np.median(o[k]))
            add = lambda o: o["add_us_row"] or float("nan")
            md = float(np.median(np.abs(np.array(a["mu"]) - np.array(b["mu"]))))
            print(
                f"{r},{n},{np.median(ca):.1f},{np.median(cb):.1f},{np.median(cb) / np.median(ca):.3f},{p:.2g},"
                f"{f(a, 'rd') / 4096:.0f},{f(b, 'rd') / 4096:.0f},{f(a, 'warm'):.0f},{f(b, 'warm'):.0f},"
                f"{add(a):.2f},{add(b):.2f},{md:.3g}"
            )


if __name__ == "__main__":
    main()
