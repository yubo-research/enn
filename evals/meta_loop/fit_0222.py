import glob
import json
import pathlib
import sys

import numpy as np


def fit(N, t, lo):
    m = N >= lo
    X = np.c_[np.ones(m.sum()), np.log(N[m])]
    c, *_ = np.linalg.lstsq(X, t[m], rcond=None)
    res = t[m] - X @ c
    se = np.sqrt(np.diag(res @ res / (m.sum() - 2) * np.linalg.inv(X.T @ X)))
    r2 = 1 - (res @ res) / ((t[m] - t[m].mean()) ** 2).sum()
    return c[0], c[1], se[1], r2


for f in sorted(glob.glob(f"{sys.argv[1]}/long_*/*.jsonl")):
    r = [json.loads(x) for x in pathlib.Path(f).read_text().splitlines()]
    N = np.array([x["N"] for x in r], float)
    print(f.split("/")[-2], f"final N={N.max():.2e}")
    for key in ("us_per_row", "query_us"):
        t = np.array([x[key] for x in r])
        for lo in (1e6, 1e7):
            a, b, se, r2 = fit(N, t, lo)
            print(
                f"  {key} N>={lo:.0e}: a={a:.1f} b={b:.2f}+/-{se:.2f} per lnN R2={r2:.2f}"
            )
        for lo, hi in [(0, 1e6), (1e6, 1e7), (1e7, 1e8 + 1)]:
            m = (N > lo) & (N <= hi)
            print(f"    ({lo:.0e},{hi:.0e}] mean {t[m].mean():.1f}")
