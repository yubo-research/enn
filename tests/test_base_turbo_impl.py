from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.turbo_config import (
    LHDOnlyConfig,
    TurboZeroConfig,
)
from enn.turbo.impl_helpers import create_trust_region
from enn.turbo.lhd_only_impl import LHDOnlyImpl
from enn.turbo.turbo_zero_impl import TurboZeroImpl


def test_turbo_zero_impl_init():
    config = TurboZeroConfig()
    impl = TurboZeroImpl(config)
    assert impl._config is config


def test_turbo_zero_impl_get_x_center():
    config = TurboZeroConfig()
    impl = TurboZeroImpl(config)
    rng = np.random.default_rng(42)
    x_obs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    y_obs = [1.0, 3.0, 2.0]
    x_center = impl.get_x_center(x_obs, y_obs, rng)
    assert x_center is not None and np.allclose(x_center, [0.3, 0.4])


def test_turbo_zero_impl_get_x_center_empty():
    config = TurboZeroConfig()
    impl = TurboZeroImpl(config)
    rng = np.random.default_rng(42)
    x_center = impl.get_x_center([], [], rng)
    assert x_center is None


def test_turbo_zero_impl_needs_tr_list():
    config = TurboZeroConfig()
    impl = TurboZeroImpl(config)
    assert not impl.needs_tr_list()


def test_create_trust_region_none():
    config = TurboZeroConfig(tr_type="none")
    rng = np.random.default_rng(42)
    tr = create_trust_region(config, 3, 4, rng)
    assert tr is not None and tr.num_dim == 3


def test_create_trust_region_turbo():
    config = TurboZeroConfig(tr_type="turbo")
    rng = np.random.default_rng(42)
    tr = create_trust_region(config, 3, 4, rng)
    assert tr is not None


def test_create_trust_region_morbo():
    config = TurboZeroConfig(tr_type="morbo", num_metrics=2)
    rng = np.random.default_rng(42)
    tr = create_trust_region(config, 3, 4, rng, num_metrics=2)
    assert tr is not None


def test_turbo_zero_impl_try_early_ask():
    config = TurboZeroConfig()
    impl = TurboZeroImpl(config)
    result = impl.try_early_ask(
        4, [], lambda n: np.zeros((n, 2)), lambda n: np.zeros((n, 2))
    )
    assert result is None


def test_handle_restart_check_morbo_no_clear():
    from enn.turbo.impl_helpers import handle_restart_check_morbo

    config = TurboZeroConfig()
    should_reset, idx = handle_restart_check_morbo(config, [], [], [], 5)
    assert not should_reset and idx == 5


def test_handle_restart_check_morbo_clears():
    from enn.turbo.impl_helpers import handle_restart_check_morbo

    config = TurboZeroConfig(tr_type="morbo")
    x, y, yvar = [1], [2], [3]
    should_reset, idx = handle_restart_check_morbo(config, x, y, yvar, 5)
    assert should_reset and idx == 0 and len(x) == 0


def test_handle_restart_clear_always():
    from enn.turbo.impl_helpers import handle_restart_clear_always

    x, y, yvar = [1], [2], [3]
    should_reset, idx = handle_restart_clear_always(x, y, yvar)
    assert should_reset and idx == 0 and len(x) == 0


@pytest.mark.parametrize(
    "y_input,expected_shape",
    [
        (np.array([1.0, 2.0]), (2, 1)),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), (2, 2)),
    ],
)
def test_turbo_zero_impl_estimate_y(y_input, expected_shape):
    config = TurboZeroConfig()
    impl = TurboZeroImpl(config)
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = impl.estimate_y(x, y_input)
    assert result.shape == expected_shape


def test_turbo_zero_impl_get_mu_sigma():
    config = TurboZeroConfig()
    impl = TurboZeroImpl(config)
    result = impl.get_mu_sigma(np.zeros((5, 2)))
    assert result is None


def test_turbo_zero_impl_select_candidates():
    config = TurboZeroConfig()
    impl = TurboZeroImpl(config)
    rng = np.random.default_rng(42)
    x_cand = np.random.rand(100, 3)
    result = impl.select_candidates(x_cand, 4, 3, rng, lambda x, n: x[:n], lambda x: x)
    assert result.shape == (4, 3)


def test_lhd_only_impl_init():
    config = LHDOnlyConfig()
    impl = LHDOnlyImpl(config)
    assert impl._config is config


def test_lhd_only_impl_select_candidates():
    config = LHDOnlyConfig()
    impl = LHDOnlyImpl(config)
    rng = np.random.default_rng(42)
    x_cand = np.random.rand(100, 3)
    result = impl.select_candidates(x_cand, 4, 3, rng, lambda x, n: x[:n], lambda x: x)
    assert result.shape == (4, 3)


def test_always_clears_on_restart_turbo_zero():
    config = TurboZeroConfig()
    impl = TurboZeroImpl(config)
    assert impl.always_clears_on_restart is False


def test_always_clears_on_restart_lhd_only():
    config = LHDOnlyConfig()
    impl = LHDOnlyImpl(config)
    assert impl.always_clears_on_restart is False


def test_always_clears_on_restart_turbo_one():
    from enn.turbo.turbo_config import TurboOneConfig
    from enn.turbo.turbo_one_impl import TurboOneImpl

    config = TurboOneConfig()
    impl = TurboOneImpl(config)
    assert impl.always_clears_on_restart is True


def test_always_clears_on_restart_turbo_enn():
    from enn.turbo.turbo_config import TurboENNConfig
    from enn.turbo.turbo_enn_impl import TurboENNImpl

    config = TurboENNConfig()
    impl = TurboENNImpl(config)
    assert impl.always_clears_on_restart is True


def test_estimate_y_passthrough():
    from enn.turbo.impl_helpers import estimate_y_passthrough

    y_1d = np.array([1.0, 2.0, 3.0])
    result = estimate_y_passthrough(y_1d)
    assert result.shape == (3, 1)

    y_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = estimate_y_passthrough(y_2d)
    assert result.shape == (2, 2)


def test_get_x_center_fallback():
    from enn.turbo.impl_helpers import get_x_center_fallback

    config = TurboZeroConfig()
    rng = np.random.default_rng(42)
    x_obs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    y_obs = [1.0, 3.0, 2.0]
    result = get_x_center_fallback(config, x_obs, y_obs, rng)
    assert result is not None
    assert np.allclose(result, [0.3, 0.4])


def test_turbo_optimizer_telemetry():
    from enn.turbo import TurboOptimizer, TurboMode

    rng = np.random.default_rng(42)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    opt = TurboOptimizer(bounds, TurboMode.TURBO_ZERO, rng=rng)
    tel = opt.telemetry()
    assert tel.dt_fit == 0.0
    assert tel.dt_sel == 0.0


def test_generate_raasp_candidates():
    from enn.turbo.turbo_utils import generate_raasp_candidates
    from scipy.stats import qmc

    rng = np.random.default_rng(42)
    center = np.array([0.5, 0.5, 0.5])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    sobol = qmc.Sobol(d=3, scramble=True, seed=42)
    result = generate_raasp_candidates(center, lb, ub, 100, rng=rng, sobol_engine=sobol)
    assert result.shape == (100, 3)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def _assert_candidates_in_unit_bounds(fn):
    rng = np.random.default_rng(42)
    center = np.array([0.5, 0.5, 0.5])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    result = fn(center, lb, ub, 100, rng=rng)
    assert result.shape == (100, 3)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_generate_raasp_candidates_uniform():
    from enn.turbo.turbo_utils import generate_raasp_candidates_uniform

    _assert_candidates_in_unit_bounds(generate_raasp_candidates_uniform)
