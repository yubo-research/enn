from __future__ import annotations

import numpy as np
import pytest

from enn import turbo_one_config
from enn.turbo.python_fallback.components.acquisition import ThompsonAcqOptimizer
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
