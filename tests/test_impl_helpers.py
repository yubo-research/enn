from __future__ import annotations
import numpy as np
from enn.turbo.config import (
    MorboTRConfig,
    MultiObjectiveConfig,
    NoTRConfig,
    TurboTRConfig,
    turbo_zero_config,
)
from enn.turbo.config.turbo_tr_config import TRLengthConfig
from enn.turbo.impl_helpers import (
    estimate_y_passthrough,
    get_x_center_fallback,
    handle_restart_check_multi_objective,
    handle_restart_clear_always,
)


def _build_and_verify_tr(tr_config, rng, num_dim=3):
    tr = tr_config.build(num_dim=num_dim, rng=rng)
    assert tr is not None
    assert hasattr(tr, "length")
    return tr


def test_trust_region_config_build_no_tr():
    _build_and_verify_tr(NoTRConfig(), np.random.default_rng(42))


def test_trust_region_config_build_turbo_tr():
    tr = _build_and_verify_tr(TurboTRConfig(), np.random.default_rng(42))
    assert tr.length == 0.8


def test_trust_region_config_build_morbo_tr():
    config = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    _build_and_verify_tr(config, np.random.default_rng(42))


def test_get_x_center_fallback_empty():
    config = turbo_zero_config()
    rng = np.random.default_rng(42)
    result = get_x_center_fallback(config, [], [], rng)
    assert result is None


def test_get_x_center_fallback_single_objective():
    config = turbo_zero_config()
    rng = np.random.default_rng(42)
    x_obs = [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]
    y_obs = [1.0, 3.0, 2.0]
    result = get_x_center_fallback(config, x_obs, y_obs, rng)
    assert result is not None
    assert np.allclose(result, [0.3, 0.7])


def test_get_x_center_fallback_with_tr_state():
    from enn.turbo.turbo_trust_region import TurboTrustRegion

    rng = np.random.default_rng(42)
    config = TurboTRConfig(
        length=TRLengthConfig(length_init=0.8, length_min=0.5**7, length_max=1.6)
    )
    tr_state = TurboTrustRegion(config=config, num_dim=2)
    x_obs = [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]
    y_obs = [1.0, 3.0, 2.0]
    result = get_x_center_fallback(None, x_obs, y_obs, rng, tr_state=tr_state)
    assert result is not None
    assert np.allclose(result, [0.3, 0.7])


def test_handle_restart_clear_always():
    x = [1, 2, 3]
    y = [4, 5, 6]
    yvar = [0.1, 0.2, 0.3]
    cleared, idx = handle_restart_clear_always(x, y, yvar)
    assert cleared is True
    assert idx == 0
    assert x == []
    assert y == []
    assert yvar == []


def test_handle_restart_check_multi_objective_single():
    from enn.turbo.turbo_trust_region import TurboTrustRegion

    config = TurboTRConfig(
        length=TRLengthConfig(length_init=0.8, length_min=0.5**7, length_max=1.6)
    )
    tr_state = TurboTrustRegion(config=config, num_dim=2)
    x = [1, 2, 3]
    y = [4, 5, 6]
    yvar = [0.1, 0.2, 0.3]
    cleared, idx = handle_restart_check_multi_objective(
        tr_state, x, y, yvar, init_idx=5
    )
    assert cleared is False
    assert idx == 5
    assert x == [1, 2, 3]


def test_handle_restart_check_multi_objective_multi():
    from enn.turbo.config.morbo_tr_config import RescalePolicyConfig
    from enn.turbo.config.rescalarize import Rescalarize
    from enn.turbo.morbo_trust_region import MorboTrustRegion

    rng = np.random.default_rng(42)
    config = MorboTRConfig(
        multi_objective=MultiObjectiveConfig(num_metrics=2),
        rescale_policy=RescalePolicyConfig(rescalarize=Rescalarize.ON_PROPOSE),
    )
    tr_state = MorboTrustRegion(config=config, num_dim=2, rng=rng)
    x = [1, 2, 3]
    y = [4, 5, 6]
    yvar = [0.1, 0.2, 0.3]
    cleared, idx = handle_restart_check_multi_objective(
        tr_state, x, y, yvar, init_idx=5
    )
    assert cleared is True
    assert idx == 0
    assert x == []


def test_estimate_y_passthrough_1d():
    y = np.array([1.0, 2.0, 3.0])
    result = estimate_y_passthrough(y)
    assert result.shape == (3, 1)


def test_estimate_y_passthrough_2d():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = estimate_y_passthrough(y)
    assert result.shape == (2, 2)
