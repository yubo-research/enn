from __future__ import annotations

import numpy as np
import pytest

from enn import create_optimizer, turbo_one_config
from enn.turbo.python_fallback.components.acquisition import (
    ThompsonAcqOptimizer,
    UCBAcqOptimizer,
)
from enn.turbo.python_fallback.components.builder import (
    build_acquisition_optimizer,
    build_surrogate,
)
from enn.turbo.python_fallback.components.protocols import (
    PosteriorResult,
    SurrogateResult,
)
from enn.turbo.python_fallback.components.surrogates import GPSurrogate
from enn.turbo.config import turbo_enn_config, turbo_zero_config
from enn.turbo.python_fallback.optimizer import Optimizer


def _make_test_data(n: int = 4, d: int = 2):
    x = np.array([[0.2, 0.3], [0.5, 0.5], [0.7, 0.8], [0.1, 0.9]])[:n, :d]
    y = np.array([0.5, 0.7, 0.3, 0.6])[:n]
    return x, y


def _make_candidates(n: int = 4, d: int = 2):
    return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])[:n, :d]


def _fit_gp_surrogate(rng):
    surrogate = GPSurrogate()
    x, y = _make_test_data()
    surrogate.fit(x, y, None, num_steps=10, rng=rng)
    return surrogate


def test_surrogate_result():
    result = SurrogateResult(model="test_model", lengthscales=np.array([1.0, 2.0]))
    assert result.model == "test_model"
    assert np.allclose(result.lengthscales, [1.0, 2.0])


def test_posterior_result():
    mu = np.array([[1.0], [2.0]])
    sigma = np.array([[0.1], [0.2]])
    result = PosteriorResult(mu=mu, sigma=sigma)
    assert np.allclose(result.mu, mu)
    assert np.allclose(result.sigma, sigma)


def test_gp_surrogate_fit_and_predict():
    rng = np.random.default_rng(42)
    surrogate = _fit_gp_surrogate(rng)
    x, _ = _make_test_data()
    posterior = surrogate.predict(x)
    assert posterior.mu.shape == (4, 1)
    assert posterior.sigma.shape == (4, 1)


def test_ucb_acq_optimizer_select():
    optimizer = UCBAcqOptimizer(beta=1.0)
    rng = np.random.default_rng(42)
    surrogate = _fit_gp_surrogate(rng)
    selected = optimizer.select(_make_candidates(), 2, surrogate, rng)
    assert selected.shape == (2, 2)


def test_build_surrogate_gp_only():
    surrogate = build_surrogate(turbo_one_config())
    assert isinstance(surrogate, GPSurrogate)


@pytest.mark.parametrize(
    "config_fn",
    [turbo_enn_config, turbo_zero_config],
)
def test_build_surrogate_rejects_rust_configs(config_fn):
    with pytest.raises(ValueError, match="Rust optimizer"):
        build_surrogate(config_fn())


def test_build_acquisition_optimizer_turbo_one():
    optimizer = build_acquisition_optimizer(turbo_one_config())
    assert isinstance(optimizer, ThompsonAcqOptimizer)


def test_optimizer_fallback_during_init():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_one_config(num_init=10, num_candidates=16)
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    x1 = opt.ask(num_arms=2)
    y1 = -np.sum(x1**2, axis=1)
    opt.tell(x1, y1)
    x2 = opt.ask(num_arms=2)
    assert x2.shape == (2, 2)
    init = opt.init_progress
    assert init is not None
    init_idx, num_init = init
    assert init_idx < num_init


def test_optimizer_direct_gp_constructor():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    config = turbo_one_config(num_init=3)
    opt = Optimizer(
        bounds=bounds,
        config=config,
        rng=rng,
        surrogate=GPSurrogate(),
        acquisition_optimizer=ThompsonAcqOptimizer(),
    )
    assert opt.init_progress == (0, 3)
