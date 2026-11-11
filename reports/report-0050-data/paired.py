import sys

import numpy as np


def ll(f):
    a = np.loadtxt(f)
    y, mu, s = a.T
    e = y - mu
    return -0.5 * np.log(2 * np.pi * s * s) - 0.5 * e * e / (s * s), e * e


def cmp(a, b, name):
    la, sa = ll(a)
    lb, sb = ll(b)
    n = len(la)
    d = lb - la
    ds = sb - sa
    print(
        f"{name}: n={n} dLL mean={d.mean():+.5f} se={d.std(ddof=1) / n**0.5:.5f} t={d.mean() / (d.std(ddof=1) / n**0.5 + 1e-30):+.2f}; dSE2 mean={ds.mean():+.6f} se={ds.std(ddof=1) / n**0.5:.6f}; rmse {np.sqrt(sa.mean()):.4f}->{np.sqrt(sb.mean()):.4f}; identical={np.array_equal(la, lb)}"
    )


base = sys.argv[1]
for other in sys.argv[2:]:
    cmp(base, other, other.split("/")[-1])
