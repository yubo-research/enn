"""noise_aware incumbent: mu ranking vs raw y; Rust config pass-through."""

from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.python_fallback.components.posterior_result import PosteriorResult
from enn.turbo.config import TurboTRConfig, turbo_one_config

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def test_fallback_noise_aware_prefers_mu_over_observed_y():
    from enn.turbo.python_fallback.optimizer import (
        create_optimizer as create_fallback_optimizer,
    )

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    config = turbo_one_config(
        trust_region=TurboTRConfig(noise_aware=True),
        num_init=2,
    )
    opt = create_fallback_optimizer(bounds=bounds, config=config, rng=rng)
    x = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]], dtype=float)
    y = np.array([[0.9], [1.0], [2.0], [0.5]], dtype=float)

    def _predict(x_in):
        mu = np.array([10.0, 0.0, 5.0, 1.0], dtype=float)[: x_in.shape[0]].reshape(
            -1, 1
        )
        return PosteriorResult(mu=mu, sigma=None)

    opt._surrogate.predict = _predict
    for i in range(x.shape[0]):
        opt._x_obs.append(x[i])
        opt._y_obs.append(y[i])
        opt._incumbent_tracker.tell(i, y[i, 0])
    opt._update_incumbent()
    assert opt._incumbent_idx == 0
    assert float(opt._incumbent_y_scalar[0]) == 10.0
    raw_y_best = int(np.argmax(y[:, 0]))
    assert raw_y_best == 2
    assert opt._incumbent_idx != raw_y_best


def test_rust_noise_aware_sets_incumbent_after_tell():
    from enn.turbo.config import ENNFitConfig, ENNSurrogateConfig, turbo_enn_config
    from .optimizer_parity_helpers import get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=5)),
        trust_region=TurboTRConfig(noise_aware=True),
        num_init=2,
    )
    opt = get_rust_optimizer(bounds, config, seed=0)
    x = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]], dtype=float)
    y = np.array([[0.0], [1.0], [2.0], [0.5]], dtype=float)
    opt.tell(x, y)
    inc = opt._inner.incumbent_x_unit()
    assert inc is not None
