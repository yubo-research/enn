"""Sanity checks: analytic gradient vs finite differences, and agreement with Rust ENN LOOCV."""

import numpy as np

from iaml_core import BpannPool, knn, loglik_grad, mean_ll

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import subsample_loglik
from enn.enn.enn_params import ENNParams

rng = np.random.default_rng(0)
n, d, k = 300, 5, 8
x = rng.random((n, d))
y = np.sin(6 * x[:, 0]) + 0.1 * rng.standard_normal(n)
y = (y - y.mean()) / y.std()
a = np.exp(rng.normal(size=d))
c = 0.07
nbr = knn(x, x, k, a, self_offset=0)

_, ga, gc = loglik_grad(x, y, x, y, nbr, a, c)
ga, gc = ga.mean(0), gc.mean()
h = 1e-6
fd = []
for i in range(d):
    ap, am = a.copy(), a.copy()
    ap[i] *= np.exp(h)
    am[i] *= np.exp(-h)
    fd.append((mean_ll(x, y, x, y, nbr, ap, c) - mean_ll(x, y, x, y, nbr, am, c)) / (2 * h))
fdc = (mean_ll(x, y, x, y, nbr, a, c * np.exp(h)) - mean_ll(x, y, x, y, nbr, a, c * np.exp(-h))) / (2 * h)
print("grad log a max abs err:", np.max(np.abs(np.array(fd) - ga)))
print("grad log c abs err:", abs(fdc - gc))

xs = x * np.sqrt(a)
m = EpistemicNearestNeighbors(xs, y.reshape(-1, 1))
p = ENNParams(k_num_neighbors=k, epistemic_variance_scale=1.0, aleatoric_variance_scale=c)
rust = subsample_loglik(m, xs, y.reshape(-1, 1), paramss=[p], P=n, rng=rng, y_std=np.ones(1))
print("rust LOOCV loglik:", rust, " mine (sum):", mean_ll(x, y, x, y, nbr, a, c) * n)

pool = BpannPool(x, y, scale=a)
bn = pool.query(x, k, exclude_self=True)
print("BPANN vs exact neighbor-set agreement:", np.mean(np.sort(bn, 1) == np.sort(nbr, 1)))
