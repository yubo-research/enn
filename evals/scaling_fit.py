"""Multiple linear regression of a metric on ln N, N and N^2, with backward elimination.

Model: ``y = b0 + b_lnN ln N + b_N N + b_N2 N^2`` fit by ordinary least squares, N in thousands
of rows (``N_UNIT``). Each coefficient gets a two-sided t test on ``obs - num_terms - 1``
residual degrees of freedom. Backward elimination starts from all three terms and drops the one
with the largest p-value while that p-value exceeds ``alpha``; the terms left are accepted and
the dropped ones rejected. The intercept is always kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

TERMS: tuple[str, ...] = ("lnN", "N", "N2")
N_UNIT = 1000.0
ALPHA = 0.05


@dataclass(frozen=True)
class TermTest:
    coef: float
    t: float
    p: float


@dataclass(frozen=True)
class RegFit:
    intercept: float
    tests: dict[str, TermTest]
    r2: float
    obs: int


def _column(term: str, n: np.ndarray) -> np.ndarray:
    if term == "lnN":
        return np.log(n)
    if term == "N":
        return n / N_UNIT
    if term == "N2":
        return (n / N_UNIT) ** 2
    raise ValueError(f"unknown term {term!r}; expected one of {TERMS}")


def _t_test(coef: float, se: float, dof: int) -> TermTest:
    if dof <= 0 or not math.isfinite(se):
        return TermTest(coef, math.nan, math.nan)
    if se == 0.0:
        return TermTest(coef, math.copysign(math.inf, coef), 0.0) if coef else TermTest(coef, math.nan, math.nan)
    t = coef / se
    return TermTest(coef, t, float(2.0 * stats.t.sf(abs(t), dof)))


def ols(ns: list[float], ys: list[float], terms: tuple[str, ...] = TERMS) -> RegFit:
    """OLS of ``ys`` on an intercept and ``terms`` of ``ns``; non-finite ``ys`` are skipped."""
    pts = np.array([(n, y) for n, y in zip(ns, ys) if math.isfinite(y)], dtype=float).reshape(-1, 2)
    n, y = pts[:, 0], pts[:, 1]
    x = np.column_stack([np.ones_like(n), *(_column(t, n) for t in terms)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - x.shape[1]
    sigma2 = float(resid @ resid) / dof if dof > 0 else math.nan
    se = np.sqrt(np.abs(np.diag(np.linalg.pinv(x.T @ x))) * sigma2)
    tss = float(((y - y.mean()) ** 2).sum()) if len(y) else 0.0
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else math.nan
    tests = {t: _t_test(float(beta[i + 1]), float(se[i + 1]), dof) for i, t in enumerate(terms)}
    return RegFit(float(beta[0]), tests, r2, len(y))


def _worst_term(fit: RegFit) -> tuple[str, float]:
    return max(
        ((t, 1.0 if math.isnan(tt.p) else tt.p) for t, tt in fit.tests.items()),
        key=lambda item: item[1],
    )


def backward_eliminate(ns: list[float], ys: list[float], alpha: float = ALPHA) -> tuple[RegFit, RegFit]:
    """Full fit on ``TERMS`` and the reduced fit left after backward elimination at ``alpha``."""
    full = ols(ns, ys, TERMS)
    fit = full
    while fit.tests:
        term, p = _worst_term(fit)
        if p <= alpha:
            break
        fit = ols(ns, ys, tuple(t for t in fit.tests if t != term))
    return full, fit
