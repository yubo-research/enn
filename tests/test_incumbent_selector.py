from __future__ import annotations
import numpy as np
import pytest
from enn.turbo.components.incumbent_selector import (
    ChebyshevIncumbentSelector,
    NoIncumbentSelector,
    ScalarIncumbentSelector,
)


@pytest.fixture
def scalar_test_data():
    return {
        "y_obs": np.array([[1.0], [3.0], [2.0]]),
        "mu_obs": np.array([[2.5], [1.5], [2.0]]),
        "rng": np.random.default_rng(42),
    }


def test_scalar_incumbent_selector_noise_oblivious(scalar_test_data):
    selector = ScalarIncumbentSelector(noise_aware=False)
    idx = selector.select(
        scalar_test_data["y_obs"], scalar_test_data["mu_obs"], scalar_test_data["rng"]
    )
    assert idx == 1


def test_scalar_incumbent_selector_noise_aware(scalar_test_data):
    selector = ScalarIncumbentSelector(noise_aware=True)
    idx = selector.select(
        scalar_test_data["y_obs"], scalar_test_data["mu_obs"], scalar_test_data["rng"]
    )
    assert idx == 0


def test_scalar_incumbent_selector_noise_aware_requires_mu(scalar_test_data):
    selector = ScalarIncumbentSelector(noise_aware=True)
    with pytest.raises(ValueError, match="noise_aware=True requires a surrogate"):
        selector.select(scalar_test_data["y_obs"], None, scalar_test_data["rng"])


def test_scalar_incumbent_selector_reset_is_noop():
    selector = ScalarIncumbentSelector(noise_aware=True)
    selector.reset(np.random.default_rng(42))


def test_chebyshev_incumbent_selector_basic():
    selector = ChebyshevIncumbentSelector(num_metrics=2, alpha=0.05, noise_aware=False)
    rng = np.random.default_rng(42)
    selector.reset(rng)
    y_obs = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    idx = selector.select(y_obs, None, rng)
    assert 0 <= idx < 3


def test_chebyshev_incumbent_selector_noise_aware():
    selector = ChebyshevIncumbentSelector(num_metrics=2, alpha=0.05, noise_aware=True)
    rng = np.random.default_rng(42)
    selector.reset(rng)
    y_obs = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    mu_obs = np.array([[0.3, 0.7], [0.6, 0.6], [0.7, 0.3]])
    idx = selector.select(y_obs, mu_obs, rng)
    assert 0 <= idx < 3


def test_chebyshev_incumbent_selector_noise_aware_requires_mu():
    selector = ChebyshevIncumbentSelector(num_metrics=2, alpha=0.05, noise_aware=True)
    rng = np.random.default_rng(42)
    selector.reset(rng)
    y_obs = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    with pytest.raises(ValueError, match="noise_aware=True requires a surrogate"):
        selector.select(y_obs, None, rng)


def test_chebyshev_incumbent_selector_reset_resamples_weights():
    selector = ChebyshevIncumbentSelector(num_metrics=2, noise_aware=False, alpha=0.05)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(123)
    selector.reset(rng1)
    w1 = selector._weights.copy()
    selector.reset(rng2)
    w2 = selector._weights.copy()
    assert not np.allclose(w1, w2)


def test_chebyshev_incumbent_selector_invalid_num_metrics():
    import pytest

    with pytest.raises(ValueError, match="num_metrics must be >= 1"):
        ChebyshevIncumbentSelector(num_metrics=0, noise_aware=False, alpha=0.05)


def test_no_incumbent_selector_always_returns_zero(scalar_test_data):
    selector = NoIncumbentSelector()
    idx = selector.select(scalar_test_data["y_obs"], None, scalar_test_data["rng"])
    assert idx == 0


def test_no_incumbent_selector_reset_is_noop():
    selector = NoIncumbentSelector()
    selector.reset(np.random.default_rng(42))
