from __future__ import annotations
import numpy as np
import pytest
from enn.enn.enn_util import (
    calculate_sobol_indices,
    arms_from_pareto_fronts,
    pareto_front_2d_maximize,
    standardize_y,
)


def _make_sobol_synth_data(*, rng, n: int, d: int, y_2d: bool) -> tuple:
    x = rng.standard_normal((n, d))
    y = x[:, 0] + 0.1 * rng.standard_normal(n)
    if y_2d:
        y = y.reshape(-1, 1)
    return x, y


@pytest.mark.parametrize(
    "n,d,y_2d,expected_check",
    [
        (50, 3, False, lambda S: S[0] > S[1] and S[0] > S[2]),
        (50, 3, True, lambda S: np.all(S >= 0) and np.all(S <= 1)),
    ],
)
def test_calculate_sobol_indices_with_data(n, d, y_2d, expected_check):
    rng = np.random.default_rng(42)
    x, y = _make_sobol_synth_data(rng=rng, n=n, d=d, y_2d=y_2d)
    S = calculate_sobol_indices(x, y)
    assert S.shape == (d,) and np.all(S >= 0) and np.all(S <= 1) and expected_check(S)


@pytest.mark.parametrize(
    "make_data,expected_check",
    [
        (
            lambda rng: (rng.standard_normal((5, 2)), rng.standard_normal(5)),
            lambda S: np.all(S == 1.0),
        ),
        (
            lambda rng: (rng.standard_normal((50, 3)), np.ones(50)),
            lambda S: np.all(S == 1.0),
        ),
    ],
)
def test_calculate_sobol_indices_edge_cases(make_data, expected_check):
    rng = np.random.default_rng(42)
    x, y = make_data(rng)
    S = calculate_sobol_indices(x, y)
    d = x.shape[1]
    assert S.shape == (d,) and expected_check(S)


def test_calculate_sobol_indices_low_variance_dimension():
    rng = np.random.default_rng(42)
    n, d = 50, 3
    x = np.zeros((n, d))
    x[:, 0], x[:, 1], x[:, 2] = (
        rng.standard_normal(n),
        1e-15 * rng.standard_normal(n),
        rng.standard_normal(n),
    )
    y = x[:, 0] + x[:, 2] + 0.1 * rng.standard_normal(n)
    S = calculate_sobol_indices(x, y)
    assert S.shape == (d,) and S[1] == 0.0 and S[0] > 0 and S[2] > 0


def test_calculate_sobol_indices_dtype_preservation():
    rng = np.random.default_rng(42)
    n, d = 50, 3
    x, y = (
        rng.standard_normal((n, d)).astype(np.float32),
        rng.standard_normal(n).astype(np.float32),
    )
    S = calculate_sobol_indices(x, y)
    assert S.shape == (d,) and S.dtype == np.float32


def test_arms_from_pareto_fronts_selects_fronts_in_order():
    x_cand = np.arange(12, dtype=float).reshape(6, 2)
    mu, se = (
        np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
        np.array([0.10, 0.20, 0.15, 0.40, 0.05, 0.50]),
    )
    rng = np.random.default_rng(0)
    out4 = arms_from_pareto_fronts(x_cand, mu, se, num_arms=4, rng=rng)
    assert out4.shape == (4, 2) and np.allclose(out4, x_cand[[0, 1, 3, 5]])
    rng = np.random.default_rng(0)
    out5 = arms_from_pareto_fronts(x_cand, mu, se, num_arms=5, rng=rng)
    assert out5.shape == (5, 2) and np.allclose(out5, x_cand[[0, 1, 2, 3, 5]])


def test_pareto_front_2d_maximize_basic():
    a, b = np.array([1.0, 0.5, 0.2]), np.array([0.5, 1.0, 0.2])
    idx = pareto_front_2d_maximize(a, b)
    assert set(idx.tolist()) == {0, 1}


@pytest.mark.parametrize(
    "y_input,expected_center,check_scale",
    [
        (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            3.0,
            lambda s, y: np.isclose(s, np.std(y)),
        ),
        ([10.0, 20.0, 30.0], 20.0, lambda s, y: s > 0),
        (np.array([5.0, 5.0, 5.0, 5.0]), 5.0, lambda s, y: s == 1.0),
        ([42.0], 42.0, lambda s, y: s == 1.0),
    ],
)
def test_standardize_y(y_input, expected_center, check_scale):
    center, scale = standardize_y(y_input)
    assert center == expected_center and check_scale(scale, y_input)
