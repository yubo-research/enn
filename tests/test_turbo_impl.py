from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.turbo_config import (
    LHDOnlyConfig,
    TurboConfig,
    TurboENNConfig,
    TurboOneConfig,
)
from enn.turbo.turbo_enn_impl import TurboENNImpl
from enn.turbo.turbo_mode import TurboMode
from enn.turbo.turbo_one_impl import TurboOneImpl
from enn.turbo.turbo_optimizer import TurboOptimizer


def test_lhd_only_runs_after_initial_observations():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    config = LHDOnlyConfig(num_init=4)
    opt = TurboOptimizer(bounds=bounds, mode=TurboMode.LHD_ONLY, rng=rng, config=config)
    for _ in range(3):
        x = opt.ask(num_arms=2)
        y = -np.sum(x**2, axis=1)
        opt.tell(x, y)
    x_after_init = opt.ask(num_arms=2)
    assert x_after_init.shape == (2, 2)


def test_turbo_enn_impl_get_x_center():
    rng = np.random.default_rng(42)
    config = TurboENNConfig(k=5)
    impl = TurboENNImpl(config)
    assert impl.get_x_center([], [], rng) is None

    x_obs = rng.random((20, 3)).tolist()
    y_obs = [float(i) for i in range(20)]
    result_before_fit = impl.get_x_center(x_obs, y_obs, rng)
    assert result_before_fit is not None
    assert np.allclose(result_before_fit, np.asarray(x_obs, dtype=float)[19])

    impl.prepare_ask(x_obs, y_obs, [0.0] * 20, num_dim=3, gp_num_steps=10, rng=rng)
    result_after_fit = impl.get_x_center(x_obs, y_obs, rng)
    assert result_after_fit is not None and result_after_fit.shape == (3,)
    x_array = np.asarray(x_obs, dtype=float)
    top_5_indices = np.argsort(y_obs)[-5:]
    assert any(
        np.allclose(result_after_fit, x_array[top_5_indices[i]]) for i in range(5)
    )


def test_turbo_one_impl_get_x_center_requires_fit_for_n_ge_2():
    rng = np.random.default_rng(0)
    impl = TurboOneImpl(TurboOneConfig())
    x1 = rng.random((1, 3)).tolist()
    center1 = impl.get_x_center(x1, [1.0], rng)
    assert center1 is not None and np.allclose(center1, np.asarray(x1, dtype=float)[0])

    x2 = rng.random((2, 3)).tolist()
    with pytest.raises(RuntimeError, match="prepare_ask"):
        impl.get_x_center(x2, [0.0, 1.0], rng)

    impl.prepare_ask(x2, [0.0, 1.0], [], num_dim=3, gp_num_steps=5, rng=rng)
    center2 = impl.get_x_center(x2, [0.0, 1.0], rng)
    assert center2 is not None
    x2_array = np.asarray(x2, dtype=float)
    assert np.allclose(center2, x2_array[0]) or np.allclose(center2, x2_array[1])


def test_turbo_config_num_metrics_validation():
    TurboConfig(tr_type="turbo", num_metrics=None)
    TurboConfig(tr_type="turbo", num_metrics=1)
    with pytest.raises(ValueError, match="num_metrics must be 1 for tr_type='turbo'"):
        TurboConfig(tr_type="turbo", num_metrics=2)
    TurboConfig(tr_type="none", num_metrics=None)
    TurboConfig(tr_type="none", num_metrics=1)
    with pytest.raises(ValueError, match="num_metrics must be 1 for tr_type='none'"):
        TurboConfig(tr_type="none", num_metrics=2)
    TurboConfig(tr_type="morbo", num_metrics=None)
    TurboConfig(tr_type="morbo", num_metrics=2)
    with pytest.raises(ValueError, match="num_metrics must be >= 1"):
        TurboConfig(tr_type="morbo", num_metrics=0)
