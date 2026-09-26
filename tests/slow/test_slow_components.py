from __future__ import annotations

import numpy as np
import pytest

from enn import create_optimizer, turbo_one_config
from enn.turbo.python_fallback.components.acquisition import UCBAcqOptimizer
from enn.turbo.python_fallback.components.surrogates import GPSurrogate


def _make_test_data(n: int = 4, d: int = 2):
    x = np.array([[0.2, 0.3], [0.5, 0.5], [0.7, 0.8], [0.1, 0.9]])[:n, :d]
    y = np.array([0.5, 0.7, 0.3, 0.6])[:n]
    return x, y


def _make_candidates(n: int = 4, d: int = 2):
    return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])[:n, :d]


def _fit_gp_surrogate(rng):
    surrogate = GPSurrogate()
    x, y = _make_test_data()
    surrogate.fit(x, y, None, num_steps=2, rng=rng)
    return surrogate


@pytest.mark.slow
def test_gp_surrogate_fit_and_predict():
    rng = np.random.default_rng(42)
    surrogate = _fit_gp_surrogate(rng)
    x, _ = _make_test_data()
    posterior = surrogate.predict(x)
    assert posterior.mu.shape == (4, 1)
    assert posterior.sigma.shape == (4, 1)


@pytest.mark.slow
def test_ucb_acq_optimizer_select():
    optimizer = UCBAcqOptimizer(beta=1.0)
    rng = np.random.default_rng(42)
    surrogate = _fit_gp_surrogate(rng)
    selected = optimizer.select(_make_candidates(), 2, surrogate, rng)
    assert selected.shape == (2, 2)


@pytest.mark.slow
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
