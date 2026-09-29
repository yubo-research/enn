"""Simple linear regressions of a metric on ln N, N and N^2 in turn, and the best of them.

For each term ``f`` in ``TERMS`` the model ``y = b0 + b f(N)`` is fit by ordinary least squares,
N in thousands of rows (``N_UNIT``), and ``b`` gets a two-sided t test on ``obs - 2`` residual
degrees of freedom. Every model has the same two parameters, so the one with the largest r^2
(equivalently the smallest residual sum of squares, or AIC) is the best fit. If even the best
term's slope is not significant at ``alpha`` the best fit is reported as ``none`` (constant in N).
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


def single_term_fits(ns: list[float], ys: list[float]) -> dict[str, RegFit]:
    """One simple regression ``y = b0 + b f(N)`` per term ``f`` in ``TERMS``."""
    return {t: ols(ns, ys, (t,)) for t in TERMS}


def best_term(fits: dict[str, RegFit], alpha: float = ALPHA) -> str:
    """Term with the largest r^2, or ``"none"`` if its slope's p-value exceeds ``alpha`` (or is nan)."""
    scored = [(t, f) for t, f in fits.items() if math.isfinite(f.r2)]
    if not scored:
        return "none"
    term, fit = max(scored, key=lambda item: item[1].r2)
    p = fit.tests[term].p
    return term if math.isfinite(p) and p <= alpha else "none"
