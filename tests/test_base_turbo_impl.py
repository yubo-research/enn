from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.base_turbo_impl import BaseTurboImpl
from enn.turbo.turbo_config import TurboConfig, TurboZeroConfig
from enn.turbo.turbo_zero_impl import TurboZeroImpl
from enn.turbo.lhd_only_impl import LHDOnlyImpl
from enn.turbo.turbo_config import LHDOnlyConfig


def test_base_turbo_impl_init():
    config = TurboConfig()
    impl = BaseTurboImpl(config)
    assert impl._config is config


def test_base_turbo_impl_get_x_center():
    config = TurboConfig()
    impl = BaseTurboImpl(config)
    rng = np.random.default_rng(42)
    x_obs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    y_obs = [1.0, 3.0, 2.0]
    x_center = impl.get_x_center(x_obs, y_obs, rng)
    assert x_center is not None and np.allclose(x_center, [0.3, 0.4])


def test_base_turbo_impl_get_x_center_empty():
    config = TurboConfig()
    impl = BaseTurboImpl(config)
    rng = np.random.default_rng(42)
    x_center = impl.get_x_center([], [], rng)
    assert x_center is None


def test_base_turbo_impl_needs_tr_list():
    config = TurboConfig()
    impl = BaseTurboImpl(config)
    assert not impl.needs_tr_list()


def test_base_turbo_impl_create_trust_region_none():
    config = TurboConfig(tr_type="none")
    impl = BaseTurboImpl(config)
    rng = np.random.default_rng(42)
    tr = impl.create_trust_region(3, 4, rng)
    assert tr is not None and tr.num_dim == 3


def test_base_turbo_impl_create_trust_region_turbo():
    config = TurboConfig(tr_type="turbo")
    impl = BaseTurboImpl(config)
    rng = np.random.default_rng(42)
    tr = impl.create_trust_region(3, 4, rng)
    assert tr is not None


def test_base_turbo_impl_create_trust_region_morbo():
    config = TurboConfig(tr_type="morbo", num_metrics=2)
    impl = BaseTurboImpl(config)
    rng = np.random.default_rng(42)
    tr = impl.create_trust_region(3, 4, rng, num_metrics=2)
    assert tr is not None


def test_base_turbo_impl_try_early_ask():
    config = TurboConfig()
    impl = BaseTurboImpl(config)
    result = impl.try_early_ask(
        4, [], lambda n: np.zeros((n, 2)), lambda n: np.zeros((n, 2))
    )
    assert result is None


def test_base_turbo_impl_handle_restart():
    config = TurboConfig()
    impl = BaseTurboImpl(config)
    should_reset, idx = impl.handle_restart([], [], [], 5, 10)
    assert not should_reset and idx == 5


def test_base_turbo_impl_handle_restart_morbo():
    config = TurboConfig(tr_type="morbo")
    impl = BaseTurboImpl(config)
    x, y, yvar = [1], [2], [3]
    should_reset, idx = impl.handle_restart(x, y, yvar, 5, 10)
    assert should_reset and idx == 0 and len(x) == 0


def test_base_turbo_impl_estimate_y_1d():
    config = TurboConfig()
    impl = BaseTurboImpl(config)
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    y = np.array([1.0, 2.0])
    result = impl.estimate_y(x, y)
    assert result.shape == (2, 1)


def test_base_turbo_impl_estimate_y_2d():
    config = TurboConfig()
    impl = BaseTurboImpl(config)
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = impl.estimate_y(x, y)
    assert result.shape == (2, 2)


def test_base_turbo_impl_get_mu_sigma():
    config = TurboConfig()
    impl = BaseTurboImpl(config)
    result = impl.get_mu_sigma(np.zeros((5, 2)))
    assert result is None


def test_base_turbo_impl_select_candidates_not_implemented():
    config = TurboConfig()
    impl = BaseTurboImpl(config)
    rng = np.random.default_rng(42)
    with pytest.raises(NotImplementedError):
        impl.select_candidates(
            np.zeros((10, 2)), 4, 2, rng, lambda x, n: x[:n], lambda x: x
        )


def test_turbo_zero_impl_init():
    config = TurboZeroConfig()
    impl = TurboZeroImpl(config)
    assert impl._config is config


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
