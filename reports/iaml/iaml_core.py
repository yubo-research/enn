"""Incremental, approximate metric learning (IAML) for ENN.

The ENN predictive model reproduced here matches the Rust implementation
(rust/crates/ennbo/src/posterior.rs, fit.rs) with observation_noise=True, no yvar,
and epistemic_variance_scale fixed at 1 (it is absorbed into the metric weights a):

    d2_j = sum_d a_d (x_d - x_jd)^2
    w_j  = 1 / (eps + d2_j + c)
    mu   = sum_j w_j y_j / sum_j w_j
    var  = 1 / sum_j w_j + c
"""

from __future__ import annotations

import tempfile

import numpy as np
from scipy.optimize import minimize

EPS = 1e-9
LOG_BOUND = 12.0


def knn(xq, xc, k, a, self_offset=None, chunk=1024):
    """Exact top-k under metric a. If self_offset is given, query i excludes row self_offset+i."""
    s = np.sqrt(a)
    qs, cs = xq * s, xc * s
    cc = (cs * cs).sum(1)
    out = np.empty((len(xq), k), dtype=np.int64)
    for lo in range(0, len(xq), chunk):
        q = qs[lo : lo + chunk]
        d2 = (q * q).sum(1)[:, None] + cc[None, :] - 2.0 * q @ cs.T
        if self_offset is not None:
            rows = np.arange(len(q))
            d2[rows, self_offset + lo + rows] = np.inf
        kk = min(k, d2.shape[1] - 1)
        part = np.argpartition(d2, kk, axis=1)[:, :k]
        out[lo : lo + chunk] = part
    return out


class BpannPool:
    """Raw-coordinate (or fixed-metric) BPANN index used as a candidate generator."""

    def __init__(self, x, y, scale=None):
        from enn.enn.enn_class import EpistemicNearestNeighbors
        from enn.turbo.config.enn_index_driver import ENNIndexDriver

        self.scale = np.ones(x.shape[1]) if scale is None else np.sqrt(scale)
        self.model = EpistemicNearestNeighbors(
            x * self.scale,
            y.reshape(-1, 1),
            index_driver=ENNIndexDriver.BPANN_DISK,
            work_dir=tempfile.mkdtemp(prefix="iaml_"),
        )
        self.rows_indexed = len(x)

    def add(self, x, y):
        self.model.add(x * self.scale, y.reshape(-1, 1))
        self.rows_indexed += len(x)

    def query(self, xq, k, exclude_self):
        from enn.enn.enn_params import ENNParams, PosteriorFlags

        k = min(k, len(self.model) - (1 if exclude_self else 0))
        p = ENNParams(k_num_neighbors=k, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1)
        flags = PosteriorFlags(exclude_nearest=exclude_self)
        return np.asarray(self.model.batch_posterior(xq * self.scale, [p], flags=flags).idx, dtype=np.int64)


def rerank(xq, xc, pool, k, a):
    """Choose the k members of each candidate pool that are nearest under metric a."""
    d2 = ((xq[:, None, :] - xc[pool]) ** 2) @ a
    kk = min(k, pool.shape[1])
    sel = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
    return np.take_along_axis(pool, sel, axis=1)


def loglik_grad(xq, yq, xc, yc, nbr, a, c):
    """Per-query Gaussian log-likelihood and gradient w.r.t. (log a, log c) with fixed neighbors."""
    diff2 = (xq[:, None, :] - xc[nbr]) ** 2
    ysel = yc[nbr]
    v = EPS + diff2 @ a + c
    w = 1.0 / v
    s = w.sum(1)
    mu = (w * ysel).sum(1) / s
    var = 1.0 / s + c
    r = yq - mu
    ll = -0.5 * np.log(2 * np.pi * var) - 0.5 * r * r / var
    gmu = r / var
    gvar = -0.5 / var + 0.5 * r * r / (var * var)
    g2 = w * w
    dmu_da = -np.einsum("nk,nkd->nd", g2 * (ysel - mu[:, None]), diff2) / s[:, None]
    dvar_da = np.einsum("nk,nkd->nd", g2, diff2) / (s * s)[:, None]
    dl_dloga = (gmu[:, None] * dmu_da + gvar[:, None] * dvar_da) * a
    dmu_dc = -(g2 * (ysel - mu[:, None])).sum(1) / s
    dvar_dc = g2.sum(1) / (s * s) + 1.0
    dl_dlogc = (gmu * dmu_dc + gvar * dvar_dc) * c
    return ll, dl_dloga, dl_dlogc


def mean_ll(xq, yq, xc, yc, nbr, a, c):
    return float(loglik_grad(xq, yq, xc, yc, nbr, a, c)[0].mean())


class Metric:
    """Parameters theta = (log a, log c). If isotropic, a single log a is shared by all dims."""

    def __init__(self, d, isotropic=False, log_a0=0.0, log_c0=np.log(0.1)):
        self.d = d
        self.iso = isotropic
        self.theta = np.concatenate([np.full(1 if isotropic else d, log_a0), [log_c0]])

    def a_of(self, theta):
        la = theta[:-1]
        return np.exp(np.full(self.d, la[0]) if self.iso else la)

    @property
    def a(self):
        return self.a_of(self.theta)

    @property
    def c(self):
        return float(np.exp(self.theta[-1]))

    def grad_map(self, g_loga, g_logc):
        ga = g_loga.sum(1, keepdims=True) if self.iso else g_loga
        return np.concatenate([ga, g_logc[:, None]], axis=1)

    def fit_fixed(self, xq, yq, xc, yc, nbr, maxiter=40):
        def f(theta):
            a, c = self.a_of(theta), float(np.exp(theta[-1]))
            ll, ga, gc = loglik_grad(xq, yq, xc, yc, nbr, a, c)
            g = self.grad_map(ga, gc).mean(0)
            return -ll.mean(), -g

        res = minimize(
            f,
            self.theta,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-LOG_BOUND, LOG_BOUND)] * len(self.theta),
            options={"maxiter": maxiter},
        )
        self.theta = res.x
        return res

    def sgd_step(self, xq, yq, xc, yc, nbr, opt):
        ll, ga, gc = loglik_grad(xq, yq, xc, yc, nbr, self.a, self.c)
        g = self.grad_map(ga, gc).mean(0)
        self.theta = np.clip(opt.step(self.theta, -g), -LOG_BOUND, LOG_BOUND)
        return ll


class Adam:
    def __init__(self, n, lr=0.05, b1=0.9, b2=0.999):
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0
        self.lr, self.b1, self.b2 = lr, b1, b2

    def step(self, theta, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return theta - self.lr * mh / (np.sqrt(vh) + 1e-8)


def _fit_from(metric, theta0, x, y, select, outer):
    """Alternate (select neighbors, fit with them fixed); accept a round only if true LOOCV improves."""
    metric.theta = theta0.copy()
    nbr = select(metric.a)
    best = mean_ll(x, y, x, y, nbr, metric.a, metric.c)
    for _ in range(outer):
        keep = metric.theta.copy()
        metric.fit_fixed(x, y, x, y, nbr)
        new_nbr = select(metric.a)
        now = mean_ll(x, y, x, y, new_nbr, metric.a, metric.c)
        if now <= best + 1e-6:
            metric.theta = keep
            break
        best, nbr = now, new_nbr
    return metric.theta.copy(), best


def _isotropic_start(metric, x, y, select):
    iso = Metric(metric.d, isotropic=True)
    iso.theta = np.array([np.median(metric.theta[:-1]), metric.theta[-1]])
    iso.fit_fixed(x, y, x, y, select(iso.a))
    return np.concatenate([np.full(metric.d, iso.theta[0]), [iso.theta[-1]]])


def _fit_safeguarded(metric, x, y, select, outer, restart):
    if metric.iso:
        metric.fit_fixed(x, y, x, y, select(metric.a))
        return metric
    starts = [metric.theta.copy()]
    if restart:
        starts.append(_isotropic_start(metric, x, y, select))
    results = [_fit_from(metric, s, x, y, select, outer) for s in starts]
    metric.theta = max(results, key=lambda r: r[1])[0]
    return metric


def fit_exact(metric, x, y, k, outer=4, restart=True):
    """Full-batch LOOCV fit; neighbors re-selected exactly under the current metric each round."""
    return _fit_safeguarded(metric, x, y, lambda a: knn(x, x, k, a, self_offset=0), outer, restart)


def fit_on_pools(metric, x, y, pool, k, outer=4, restart=True):
    """Full-batch LOOCV fit where each point's neighbors must come from a fixed candidate pool."""
    return _fit_safeguarded(metric, x, y, lambda a: rerank(x, x, pool, k, a), outer, restart)
