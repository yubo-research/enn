from __future__ import annotations

import math

import numpy as np
import pytest

from evals import scaling_fit as mod

NS = [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0] * 3


def _noisy(fn, seed: int = 0, sd: float = 0.01) -> list[float]:
    rng = np.random.default_rng(seed)
    return [fn(n) + sd * rng.standard_normal() for n in NS]


def test_ols_exact_coefficients() -> None:
    ys = [2.0 + 0.5 * math.log(n) + 0.1 * n / 1000 - 0.001 * (n / 1000) ** 2 for n in NS]
    fit = mod.ols(NS, ys)
    assert fit.intercept == pytest.approx(2.0)
    assert fit.tests["lnN"].coef == pytest.approx(0.5)
    assert fit.tests["N"].coef == pytest.approx(0.1)
    assert fit.tests["N2"].coef == pytest.approx(-0.001)
    assert fit.obs == len(NS) and fit.r2 == pytest.approx(1.0)


def test_ols_skips_non_finite_and_handles_zero_dof() -> None:
    fit = mod.ols([10.0, 100.0, 1000.0, 5000.0, 7.0], [1.0, 2.0, 3.0, 4.0, math.nan])
    assert fit.obs == 4
    assert all(math.isnan(tt.p) for tt in fit.tests.values())


def test_t_test_edge_cases() -> None:
    assert mod._t_test(1.0, 0.0, 5) == mod.TermTest(1.0, math.inf, 0.0)
    assert math.isnan(mod._t_test(0.0, 0.0, 5).p)
    assert math.isnan(mod._t_test(1.0, math.nan, 5).t)
    assert mod._t_test(2.0, 1.0, 1000).p == pytest.approx(0.0457, abs=1e-3)


def test_unknown_term_raises() -> None:
    with pytest.raises(ValueError, match="unknown term"):
        mod.ols(NS, NS, ("sqrtN",))


@pytest.mark.parametrize(
    ("fn", "best"),
    [
        (lambda n: 5.0 + 0.3 * n / 1000, "N"),
        (lambda n: 5.0 + 0.8 * math.log(n), "lnN"),
        (lambda n: 5.0 + 0.002 * (n / 1000) ** 2, "N2"),
        (lambda n: 5.0, "none"),
    ],
)
def test_single_term_fits_pick_true_term(fn, best: str) -> None:
    fits = mod.single_term_fits(NS, _noisy(fn))
    assert list(fits) == list(mod.TERMS)
    assert all(list(f.tests) == [t] and f.obs == len(NS) for t, f in fits.items())
    assert mod.best_term(fits) == best


def test_single_term_fit_coefficients() -> None:
    fits = mod.single_term_fits(NS, [1.0 + 0.3 * n / 1000 for n in NS])
    assert fits["N"].intercept == pytest.approx(1.0)
    assert fits["N"].tests["N"].coef == pytest.approx(0.3)
    assert fits["N"].r2 == pytest.approx(1.0)
    assert fits["lnN"].r2 < fits["N"].r2 and fits["N2"].r2 < fits["N"].r2


def test_best_term_degenerate_fits_are_none() -> None:
    flat = mod.single_term_fits(NS, [2.0] * len(NS))
    assert all(math.isnan(f.r2) for f in flat.values())
    assert mod.best_term(flat) == "none"
    assert mod.best_term({}) == "none"
