import glob
import json
from pathlib import Path

import numpy as np

R = {}
for f in sorted(glob.glob("q54_*.jsonl")):
    for l in Path(f).read_text().splitlines():
        r = json.loads(l)
        R.setdefault(r["n"], []).append(np.median(r["q1"]) * 1e3)
ns = np.array(sorted(R))
m = np.array([np.mean(R[n]) for n in ns])
se = np.array([np.std(R[n], ddof=1) / np.sqrt(3) for n in ns])
print("n  mean_ms  se  | per-seed")
for n, a, s in zip(ns, m, se):
    print(n, round(a, 2), round(s, 2), [round(x, 1) for x in R[n]])
L = np.log(ns)


def fit(mask, name):
    A = np.c_[np.ones(mask.sum()), L[mask]]
    c, _res, *_ = np.linalg.lstsq(A, m[mask], rcond=None)
    r = m[mask] - A @ c
    print(name, f"a={c[0]:.2f} b={c[1]:.2f} rmse={np.sqrt(np.mean(r**2)):.2f}")


fit(ns >= 0, "all")
fit(ns >= 10000, "N>=1e4")
fit(ns <= 30000, "N<=3e4")
for i in range(len(ns) - 1):
    print(
        f"slope ms per ln-unit {ns[i]}->{ns[i + 1]}: {(m[i + 1] - m[i]) / (L[i + 1] - L[i]):.2f}"
    )
p = np.polyfit(L[ns >= 10000], np.log(m[ns >= 10000]), 1)
print(f"power-law exponent N>=1e4: {p[0]:.3f}")
