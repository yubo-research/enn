from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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


def _beta_cf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for num in (
            m * (b - m) * x / ((a + 2 * m - 1.0) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1.0)),
        ):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def regularized_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0 if x <= 0.0 else 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_cf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_cf(b, a, 1.0 - x) / b


def two_sided_t_p(t: float, dof: int) -> float:
    if math.isinf(t):
        return 0.0
    return regularized_beta(0.5 * dof, 0.5, dof / (dof + t * t))


def _t_test(coef: float, se: float, dof: int) -> TermTest:
    if dof <= 0 or not math.isfinite(se):
        return TermTest(coef, math.nan, math.nan)
    if se == 0.0:
        return (
            TermTest(coef, math.copysign(math.inf, coef), 0.0)
            if coef
            else TermTest(coef, math.nan, math.nan)
        )
    t = coef / se
    return TermTest(coef, t, two_sided_t_p(t, dof))


def ols(ns: list[float], ys: list[float], terms: tuple[str, ...] = TERMS) -> RegFit:
    pts = np.array(
        [(n, y) for n, y in zip(ns, ys) if math.isfinite(y)], dtype=float
    ).reshape(-1, 2)
    n, y = pts[:, 0], pts[:, 1]
    x = np.column_stack([np.ones_like(n), *(_column(t, n) for t in terms)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - x.shape[1]
    sigma2 = float(resid @ resid) / dof if dof > 0 else math.nan
    se = np.sqrt(np.abs(np.diag(np.linalg.pinv(x.T @ x))) * sigma2)
    tss = float(((y - y.mean()) ** 2).sum()) if len(y) else 0.0
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else math.nan
    tests = {
        t: _t_test(float(beta[i + 1]), float(se[i + 1]), dof)
        for i, t in enumerate(terms)
    }
    return RegFit(float(beta[0]), tests, r2, len(y))


def single_term_fits(ns: list[float], ys: list[float]) -> dict[str, RegFit]:
    return {t: ols(ns, ys, (t,)) for t in TERMS}


def best_term(fits: dict[str, RegFit], alpha: float = ALPHA) -> str:
    scored = [(t, f) for t, f in fits.items() if math.isfinite(f.r2)]
    if not scored:
        return "none"
    term, fit = max(scored, key=lambda item: item[1].r2)
    p = fit.tests[term].p
    return term if math.isfinite(p) and p <= alpha else "none"
