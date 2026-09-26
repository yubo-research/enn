"""Tests for bound-aware ENNNormal.sample and confidence_interval."""

from __future__ import annotations

import numpy as np
import pytest

from enn import EpistemicNearestNeighbors
from enn.enn.enn_normal import ENNNormal
from enn.enn.enn_params import ENNParams
from ops.qa import frac_out_of_open_interval, make_bounded_1d_xy, y_bounds_array


def _assert_in_open_interval(values: np.ndarray, lo: float, hi: float) -> None:
    assert frac_out_of_open_interval(values, lo, hi) == 0.0


def _logit_model() -> EpistemicNearestNeighbors:
    bounds = np.array([[0.0, 1.0]], dtype=float)
    train_x = np.array([[0.0], [1.0], [0.5]], dtype=float)
    train_y = np.array([[0.1], [0.9], [0.5]], dtype=float)
    return EpistemicNearestNeighbors(train_x, train_y, y_bounds=bounds)


@pytest.mark.parametrize(
    "lo,hi",
    [
        (0.0, np.inf),
        (0.0, 1.0),
        (-np.inf, 0.0),
        (-1.0, 1.0),
        (-3.0, 7.0),
        (-np.inf, 4.0),
        (2.0, np.inf),
    ],
)
def test_confidence_interval_stays_in_open_interval(lo, hi):
    rng = np.random.default_rng(7)
    x_train, y_train = make_bounded_1d_xy(24, rng, lo, hi, y_scale=1.5, y_center=0.0)
    x_test = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    model = EpistemicNearestNeighbors(
        x_train, y_train, y_bounds=y_bounds_array(lo, hi)
    )
    params = ENNParams(
        k_num_neighbors=3,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
    )
    post = model.posterior(x_test, params=params)
    lower, upper = post.confidence_interval(0.95)
    _assert_in_open_interval(lower, lo, hi)
    _assert_in_open_interval(upper, lo, hi)
    assert np.all(lower <= post.mu)
    assert np.all(upper >= post.mu)


def test_confidence_interval_unbounded_matches_linear():
    mu = np.array([[1.0, -2.0]], dtype=float)
    se = np.array([[0.2, 0.5]], dtype=float)
    post = ENNNormal(mu=mu, se=se, se_epi=se.copy(), se_ale=np.zeros_like(se))
    lower, upper = post.confidence_interval(0.95)
    from enn.enn.enn_normal import _z_crit

    z_impl = _z_crit(0.95)
    np.testing.assert_allclose(lower, mu - z_impl * se)
    np.testing.assert_allclose(upper, mu + z_impl * se)


def test_sample_and_function_draw_respect_y_bounds():
    lo, hi = -3.0, 7.0
    rng = np.random.default_rng(11)
    x_train, y_train = make_bounded_1d_xy(
        16, rng, lo, hi, y_scale=2.0, y_center=-1.0
    )
    x_test = np.linspace(0.0, 1.0, 80).reshape(-1, 1)
    model = EpistemicNearestNeighbors(
        x_train, y_train, y_bounds=y_bounds_array(lo, hi)
    )
    params = ENNParams(
        k_num_neighbors=4,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.05,
    )
    post = model.posterior(x_test, params=params)
    samples = post.sample(128, np.random.default_rng(0))
    _assert_in_open_interval(samples, lo, hi)

    draws, _ = model.posterior_function_draw(
        x_test, params, function_seeds=list(range(32))
    )
    _assert_in_open_interval(draws, lo, hi)


def test_confidence_interval_on_logit_stays_inside_bounds():
    model = _logit_model()
    params = ENNParams(
        k_num_neighbors=2,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )
    x_test = np.linspace(0.001, 0.999, 30).reshape(-1, 1)
    post = model.posterior(x_test, params=params)
    lower, upper = post.confidence_interval(0.95)
    _assert_in_open_interval(lower, 0.0, 1.0)
    _assert_in_open_interval(upper, 0.0, 1.0)
    assert np.all(lower <= post.mu)
    assert np.all(upper >= post.mu)
