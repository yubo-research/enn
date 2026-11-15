import json
import sys

import numpy as np

D, M = int(sys.argv[1]), int(sys.argv[2])
NS = [int(float(a)) for a in sys.argv[3:]]
rng0 = np.random.default_rng(12345)
B = np.linalg.qr(rng0.standard_normal((D, M)))[0].T if M else None


def draw(rng, n):
    if not M:
        return rng.random((n, D)).astype(np.float32)
    return (rng.random((n, M)) @ B + 0.01 * rng.standard_normal((n, D))).astype(
        np.float32
    )


def q4(v):
    s = np.abs(v).max(1, keepdims=True) / 7.0
    s[s == 0] = 1.0
    return np.clip(np.round(v / s), -7, 7) * s


def q8(v):
    s = np.abs(v).max(1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    return np.clip(np.round(v / s), -127, 127) * s, s[:, 0]


def leaves(x, idx, cap=64):
    if len(idx) <= cap:
        return [idx]
    sub = x[idx]
    c = sub - sub.mean(0)
    w = c[np.argmax((c**2).sum(1))].copy()
    for _ in range(3):
        w = c.T @ (c @ w)
        w /= np.linalg.norm(w) + 1e-30
    p = c @ w
    o = np.argsort(p)
    h = len(idx) // 2
    return leaves(x, idx[o[:h]], cap) + leaves(x, idx[o[h:]], cap)


for n in NS:
    x = draw(np.random.default_rng(n), n)
    out = {"D": D, "intrinsic": M, "n": n}
    rad, s8, s4, sr = [], [], [], []
    for li in leaves(x, np.arange(n)):
        v = x[li]
        cen = v.mean(0)
        rad.append(np.sqrt(((v - cen) ** 2).sum(1)).max())
        _, sc = q8(v)
        s8.append(0.5 * np.sqrt(D) * sc.max())
        s4.append(np.sqrt(((v - q4(v)) ** 2).sum(1)).max())
        a, _ = q8(cen[None, :])
        r = v - a
        sr.append(np.sqrt(((r - q4(r)) ** 2).sum(1)).max())
    rad = np.array(rad)
    for k, s in (("raw8", s8), ("raw4", s4), ("res4", sr)):
        out[k + "_slack_over_radius"] = float(np.median(np.array(s) / rad))
    out["median_radius"] = float(np.median(rad))
    print(json.dumps(out), flush=True)
