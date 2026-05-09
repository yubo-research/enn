from __future__ import annotations

import numpy as np

from enn.turbo.config.morbo_tr_config import MorboTRConfig, MultiObjectiveConfig
from enn.turbo.morbo_trust_region import MorboTrustRegion


def test_morbo_resample_weights():
    rng = np.random.default_rng(42)
    config = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    tr = MorboTrustRegion(config, num_dim=3, rng=rng)
    weights_before = tr.weights.copy()
    tr.resample_weights(rng)
    weights_after = tr.weights.copy()
    assert not np.allclose(weights_before, weights_after)


def test_morbo_inner_trust_region_applies_updates_after_restart_without_intervening_validate_request():
    """Inner TurboTrustRegion must keep num_arms state across restart.

    tell() calls update() without validate_request(). After a TR restart,
    TurboHybridStrategy.ask can return init points without validate_request(), so the
    next tell must still advance inner TR state.
    """
    rng = np.random.default_rng(42)
    config = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    tr = MorboTrustRegion(config, num_dim=3, rng=rng)
    num_arms = 4
    tr.validate_request(num_arms)
    y_obs = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [0.0, 3.0]], dtype=float)
    tr.update(y_obs, tr.get_incumbent_value(y_obs, rng))
    assert tr._tr._failure_tolerance is not None
    assert tr._tr.prev_num_obs == num_arms

    tr.restart(rng)
    assert tr._tr._failure_tolerance is not None

    y_obs2 = np.array(
        [[0.5, 0.5], [0.6, 0.4], [0.4, 0.6], [0.55, 0.45]], dtype=float
    )
    tr.update(y_obs2, tr.get_incumbent_value(y_obs2, rng))
    assert tr._tr.prev_num_obs == num_arms


def test_morbo_restart_resamples():
    from enn.turbo.config import Rescalarize
    from enn.turbo.config.morbo_tr_config import RescalePolicyConfig

    rng = np.random.default_rng(42)
    config = MorboTRConfig(
        multi_objective=MultiObjectiveConfig(num_metrics=2),
        rescale_policy=RescalePolicyConfig(rescalarize=Rescalarize.ON_RESTART),
    )
    tr = MorboTrustRegion(config, num_dim=3, rng=rng)
    weights_before = tr.weights.copy()
    tr.restart(rng)
    weights_after = tr.weights.copy()
    assert not np.allclose(weights_before, weights_after)
