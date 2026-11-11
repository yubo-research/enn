import json
import sys

import numpy as np

D = 1000
N = int(sys.argv[1])
NQ = 200
kind = sys.argv[2]
A = np.random.default_rng(7).standard_normal((4, D)) / 2


def gen(n, r):
    if kind == "gauss":
        return r.standard_normal((n, D)).astype(np.float32)
    z = r.standard_normal((n, 4))
    return (z @ A + 0.1 * r.standard_normal((n, D))).astype(np.float32)


r = np.random.default_rng(1)
X = gen(N, r)
Q = gen(NQ, np.random.default_rng(2))


def d2(Q, X):
    return (Q**2).sum(1)[:, None] - 2 * Q @ X.T + (X**2).sum(1)[None, :]


gt = np.argsort(d2(Q, X), axis=1)[:, :10]
R = np.linalg.qr(np.random.default_rng(3).standard_normal((D, D)))[0].astype(np.float32)
XR = X @ R
QR = Q @ R
mu = XR.mean(0)
sd = XR.std(0) + 1e-9


def lloyd(b):
    k = 2**b
    z = np.random.default_rng(0).standard_normal(400000)
    lv = np.quantile(z, (np.arange(k) + 0.5) / k)
    for _ in range(30):
        edges = (lv[1:] + lv[:-1]) / 2
        idx = np.searchsorted(edges, z)
        lv = np.array([z[idx == i].mean() for i in range(k)])
    return lv


out = []
for b in (1, 2, 4, 8):
    lv = lloyd(b)
    edges = (lv[1:] + lv[:-1]) / 2
    codes = np.searchsorted(edges, ((XR - mu) / sd)).astype(np.uint8)
    Xh = (lv[codes] * sd + mu).astype(np.float32)
    ap = d2(QR, Xh)
    order = np.argsort(ap, axis=1)[:, :200]
    res = {"kind": kind, "N": N, "bits": b, "bytes_per_row": D * b / 8}
    for M in (10, 20, 30, 50, 100, 200):
        res[f"rec@M={M}"] = float(
            np.mean([len(set(gt[i]) & set(order[i, :M])) / 10 for i in range(NQ)])
        )
    print(json.dumps(res), flush=True)
