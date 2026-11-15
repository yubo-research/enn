import glob
import json
import pathlib
import sys

import numpy as np


def fit_ranges(N, t):
    for lo in (1e6, 1e7):
        m = N >= lo
        if m.sum() < 5:
            continue
        X = np.c_[np.ones(m.sum()), np.log(N[m])]
        c, *_ = np.linalg.lstsq(X, t[m], rcond=None)
        res = t[m] - X @ c
        se = np.sqrt(np.diag(res @ res / (m.sum() - 2) * np.linalg.inv(X.T @ X)))
        r2 = 1 - (res @ res) / ((t[m] - t[m].mean()) ** 2).sum()
        print(
            f"  fit N>={lo:.0e}: a={c[0]:.1f}, b={c[1]:.2f}+/-{se[1]:.2f} us/lnN, R2={r2:.2f}"
        )


def main():
    for f in sorted(glob.glob(f"{sys.argv[1]}/long_*/*.jsonl")):
        r = [json.loads(x) for x in pathlib.Path(f).read_text().splitlines()]
        N = np.array([x["N"] for x in r], float)
        t = np.array([x["us_per_row"] for x in r])
        rd = np.array([x["read_B_row"] for x in r])
        wr = np.array([x["write_B_row"] for x in r])
        print(f.split("/")[-2], f"final N={N.max():.2e}, total {r[-1]['t']:.0f}s")
        for lo, hi in [(0, 1e6), (1e6, 3e6), (3e6, 1e7), (1e7, 3e7), (3e7, 1e8 + 1)]:
            m = (N > lo) & (N <= hi)
            if m.sum():
                print(
                    f"  N in ({lo:.0e},{hi:.0e}]: {t[m].mean():6.1f} us/row (se {t[m].std(ddof=1) / np.sqrt(m.sum()) if m.sum() > 1 else 0:.1f}), read {rd[m].mean():5.0f} B, write {wr[m].mean():5.0f} B, n={m.sum()}"
                )
        fit_ranges(N, t)


if __name__ == "__main__":
    main()
