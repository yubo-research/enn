from __future__ import annotations
import numpy as np
from scipy.stats import qmc
from enn.turbo.config.morbo_tr_config import (
    MorboTRConfig,
    MultiObjectiveConfig,
    RescalePolicyConfig,
)
from enn.turbo.config import Rescalarize
from enn.turbo.config.turbo_tr_config import TRLengthConfig, TurboTRConfig
from enn.turbo.morbo_trust_region import MorboTrustRegion
from enn.turbo.turbo_trust_region import TurboTrustRegion


def test_trust_region_state_update_and_restart_and_bounds():
    config = TurboTRConfig(
        length=TRLengthConfig(length_init=0.8, length_min=0.5**7, length_max=1.6)
    )
    state = TurboTrustRegion(config=config, num_dim=2)
    state.validate_request(num_arms=2)
    values = []
    for v in [0.0, 1.0, 2.0]:
        values.append(v)
        y_obs = np.array(values, dtype=float)
        state.update(y_obs, np.array([float(np.max(y_obs))], dtype=float))
    x_center = np.zeros(2, dtype=float)
    lb, ub = state.compute_bounds_1d(x_center)
    assert lb.shape == (2,) and ub.shape == (2,)
    state.length = state.length_min / 2.0
    assert state.needs_restart()
    state.restart()
    assert state.length == state.length_init


def test_morbo_chebyshev_trust_region_weights_and_scaling():
    rng1, rng2 = np.random.default_rng(0), np.random.default_rng(0)
    config = MorboTRConfig(
        multi_objective=MultiObjectiveConfig(num_metrics=2),
        rescale_policy=RescalePolicyConfig(rescalarize=Rescalarize.ON_PROPOSE),
    )
    tr1 = MorboTrustRegion(config=config, num_dim=3, rng=rng1)
    tr2 = MorboTrustRegion(config=config, num_dim=3, rng=rng2)
    assert np.allclose(tr1.weights, tr2.weights)
    assert tr1.weights.shape == (2,) and np.all(tr1.weights > 0.0)
    assert np.isclose(tr1.weights.sum(), 1.0)
    y_obs = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    y_incumbent = tr1.get_incumbent_value(y_obs, rng1)
    tr1.update(y_obs, y_incumbent)
    scores = tr1.scalarize(y_obs, clip=True)
    assert scores.shape == (2,) and np.all(np.isfinite(scores))
    t = 0.5 * tr1.weights.reshape(1, -1)
    expected = np.min(t, axis=1) + 0.05 * np.sum(t, axis=1)
    assert np.allclose(scores[:1], expected)
    sobol_engine = qmc.Sobol(d=3, scramble=True, seed=0)
    x_center = np.array([0.5, 0.5, 0.5], dtype=float)
    x_cand = tr1.generate_candidates(
        x_center=x_center,
        lengthscales=None,
        num_candidates=64,
        rng=np.random.default_rng(1),
        sobol_engine=sobol_engine,
    )
    assert x_cand.shape == (64, 3) and np.all(x_cand >= 0.0) and np.all(x_cand <= 1.0)
