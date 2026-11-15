import glob
import itertools
import json
import pathlib
import sys

import numpy as np


def main():
    D = sys.argv[1]
    edges = [0, 1e6, 2.4e6, 4e6, 6e6, 9e6, 13e6, 21e6]
    for f in sorted(glob.glob(f"{D}/rep*/bs_*.jsonl")):
        r = [json.loads(x) for x in pathlib.Path(f).read_text().splitlines()]
        N = np.array([x["N"] for x in r], float)
        t = np.array([x["us_per_row"] for x in r])
        print(f)
        for lo, hi in itertools.pairwise(edges):
            m = (N > lo) & (N <= hi)
            if m.sum():
                print(
                    f"  N in ({lo:.1e},{hi:.1e}]: mean {t[m].mean():6.1f} us/row, n={m.sum()}"
                )
        for lo in (2.4e6, 4e6):
            m = N >= lo
            if m.sum() > 4:
                X = np.c_[np.ones(m.sum()), np.log(N[m])]
                coef, _res, *_ = np.linalg.lstsq(X, t[m], rcond=None)
                ss = ((t[m] - X @ coef) ** 2).sum()
                s2 = ss / (m.sum() - 2)
                se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
                print(
                    f"  fit N>={lo:.1e}: a={coef[0]:.1f}, b={coef[1]:.2f} +/- {se[1]:.2f} us per ln N; "
                    f"time at N=2.4e6 vs end: {coef[0] + coef[1] * np.log(2.4e6):.1f} -> {coef[0] + coef[1] * np.log(N.max()):.1f}"
                )


if __name__ == "__main__":
    main()
