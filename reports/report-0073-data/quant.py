import json
import sys
from pathlib import Path

import numpy as np

N, NQ, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
D, d, k = 1000, 12, 10
A = np.random.default_rng(7).standard_normal((d, D)).astype(np.float32) / 2


def gen(n, r):
    z = r.standard_normal((n, d)).astype(np.float32)
    x = z @ A + 0.1 * r.standard_normal((n, D)).astype(np.float32)
    f = np.sin(2 * z[:, 0]) + np.sin(2 * z[:, 1])
    return x, f + 0.1 * r.standard_normal(n).astype(np.float32)


r = np.random.default_rng(11)
X = np.empty((N, D), np.float32)
Y = np.empty(N, np.float32)
for i in range(0, N, 10000):
    X[i : i + 10000], Y[i : i + 10000] = gen(min(10000, N - i), r)
Q, YQ = gen(NQ, np.random.default_rng(1234))
mu, sd = X.mean(0), X.std(0)


def quant(X, b, lo, hi):
    L = 2**b - 1
    return np.clip(np.round((X - lo) / (hi - lo) * L), 0, L).astype(np.uint8), L


def dq(C, L, lo, hi):
    return C.astype(np.float32) / L * (hi - lo) + lo


def d2(Xs, q):
    return (Xs * Xs).sum(1) - 2 * Xs @ q + q @ q


x2 = (X * X).sum(1)
exact = []
for q in Q:
    dd = x2 - 2 * X @ q
    exact.append(np.argpartition(dd, k)[:k])
exact = [e[np.argsort((X[e] - q) ** 2 @ np.ones(D))] for e, q in zip(exact, Q)]
res = {"N": N, "NQ": NQ, "rows": []}
pe = np.array([Y[e].mean() for e in exact])
se_e = (pe - YQ) ** 2
res["rmse_exact"] = float(np.sqrt(se_e.mean()))
for b in (8, 6, 4, 3, 2, 1):
    for width in (3.0,):
        lo, hi = mu - width * sd, mu + width * sd
        C, L = quant(X, b, lo, hi)
        Xh = dq(C, L, lo, hi)
        xh2 = (Xh * Xh).sum(1)
        DD = [xh2 - 2 * Xh @ q for q in Q]
        for m in (10, 20, 30, 50, 100):
            rec = []
            pr = []
            for q, e, dd in zip(Q, exact, DD):
                cand = np.argpartition(dd, m)[:m]
                ex = ((X[cand] - q) ** 2).sum(1)
                top = cand[np.argsort(ex)[:k]]
                rec.append(len(set(top) & set(e)) / k)
                pr.append(Y[top].mean())
            pr = np.array(pr)
            diff = (pr - YQ) ** 2 - se_e
            row = {
                "bits": b,
                "m": m,
                "recall": float(np.mean(rec)),
                "recall_se": float(np.std(rec, ddof=1) / np.sqrt(NQ)),
                "rmse": float(np.sqrt(((pr - YQ) ** 2).mean())),
                "dsq_err_mean": float(diff.mean()),
                "dsq_err_se": float(diff.std(ddof=1) / np.sqrt(NQ)),
            }
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
Path(out).write_text(json.dumps(res))
