from __future__ import annotations
import numpy as np
from enn.turbo.morbo_trust_region import MorboTrustRegion
from enn.turbo.config.morbo_tr_config import MorboTRConfig, MultiObjectiveConfig


def test_morbo_resample_weights():
    rng = np.random.default_rng(42)
    config = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    tr = MorboTrustRegion(config, num_dim=3, rng=rng)
    weights_before = tr.weights.copy()
    tr.resample_weights(rng)
    weights_after = tr.weights.copy()
    assert not np.allclose(weights_before, weights_after)


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
