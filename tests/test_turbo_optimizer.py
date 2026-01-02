from __future__ import annotations

import numpy as np
import pytest

import conftest

from enn.turbo.turbo_config import (
    TurboConfig,
    TurboENNConfig,
    TurboOneConfig,
    TurboZeroConfig,
)
from enn.turbo.turbo_mode import TurboMode
from enn.turbo.turbo_optimizer import TurboOptimizer


def _run_bo(mode: TurboMode, num_steps: int = 15) -> float:
    from enn import Turbo

    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = Turbo(bounds=bounds, mode=mode, rng=rng)
    best = -np.inf
    for _ in range(num_steps):
        x = opt.ask(num_arms=4)
        y = conftest.sphere_objective(x)
        opt.tell(x, y)
        best = max(best, float(np.max(y)))
    return best


def test_turbo_zero_ask_tell_and_shape():
    from enn import Turbo

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = Turbo(bounds=bounds, mode=TurboMode.TURBO_ZERO, rng=rng)
    x0 = opt.ask(num_arms=4)
    assert x0.shape == (4, 2) and np.all(x0 >= 0.0) and np.all(x0 <= 1.0)
    opt.tell(x0, conftest.sphere_objective(x0))
    x1 = opt.ask(num_arms=4)
    assert x1.shape == (4, 2)


def test_turbo_optimizer_accepts_list_bounds():
    from enn import Turbo

    opt = Turbo(
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        mode=TurboMode.TURBO_ZERO,
        rng=np.random.default_rng(0),
    )
    assert opt.ask(num_arms=2).shape == (2, 2)


def test_turbo_optimizer_uniform_candidates_never_calls_sobol():
    from enn import Turbo
    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_config import TurboZeroConfig
    from unittest import mock

    # If candidate_rv="uniform", TurboOptimizer should not construct a Sobol engine.
    # We enforce that by making Sobol() raise if called.
    def _sobol_raises(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("Sobol should not be constructed for candidate_rv='uniform'")

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    with mock.patch("scipy.stats.qmc.Sobol", side_effect=_sobol_raises):
        opt = Turbo(
            bounds=bounds,
            mode=TurboMode.TURBO_ZERO,
            rng=rng,
            config=TurboZeroConfig(candidate_rv="uniform"),
        )
        x0 = opt.ask(num_arms=4)
    assert x0.shape == (4, 2)


def test_turbo_optimizer_requires_mode_specific_config():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="requires TurboENNConfig"):
        TurboOptimizer(
            bounds=bounds,
            mode=TurboMode.TURBO_ENN,
            rng=np.random.default_rng(0),
            config=TurboConfig(),
        )


def test_turbo_one_improves_on_sphere():
    assert _run_bo(TurboMode.TURBO_ONE, num_steps=12) > -0.5


def test_turbo_one_with_y_var_uses_noisy_gp():
    from enn import Turbo
    from enn.turbo.turbo_gp_noisy import TurboGPNoisy

    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = Turbo(bounds=bounds, mode=TurboMode.TURBO_ONE, rng=rng)
    for _ in range(5):
        x = opt.ask(num_arms=4)
        y = conftest.sphere_objective(x)
        opt.tell(x, y, rng.uniform(0.01, 0.1, size=y.shape))
    x = opt.ask(num_arms=4)
    assert x.shape == (4, 2) and isinstance(opt._mode_impl._gp_model, TurboGPNoisy)


def test_turbo_zero_reasonable_on_sphere():
    assert _run_bo(TurboMode.TURBO_ZERO, num_steps=12) > -1.5


def test_turbo_enn_uses_enn_and_is_reasonable():
    assert _run_bo(TurboMode.TURBO_ENN, num_steps=12) > -1.5


def test_turbo_enn_with_k_none_fits_hyperparameters():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = TurboOptimizer(
        bounds=bounds, mode=TurboMode.TURBO_ENN, rng=rng, config=TurboENNConfig(k=None)
    )
    x0 = opt.ask(num_arms=4)
    assert x0.shape == (4, 2) and np.all(x0 >= 0.0) and np.all(x0 <= 1.0)
    opt.tell(x0, -np.sum(x0**2, axis=1))
    x1 = opt.ask(num_arms=4)
    assert x1.shape == (4, 2) and np.all(x1 >= 0.0) and np.all(x1 <= 1.0)


def test_turbo_enn_config_scale_x_flag_runs():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ENN,
        rng=np.random.default_rng(0),
        config=TurboENNConfig(scale_x=True),
    )
    x0 = opt.ask(num_arms=3)
    opt.tell(x0, -np.sum(x0**2, axis=1))
    assert opt.ask(num_arms=3).shape == (3, 2)


def test_turbo_optimizer_with_trailing_obs():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    for mode, cfg in [
        (TurboMode.TURBO_ONE, TurboOneConfig(trailing_obs=5)),
        (TurboMode.TURBO_ENN, TurboENNConfig(trailing_obs=5)),
    ]:
        opt = TurboOptimizer(bounds=bounds, mode=mode, rng=rng, config=cfg)
        for _ in range(10):
            x = opt.ask(num_arms=2)
            opt.tell(x, -np.sum(x**2, axis=1))
        assert len(opt._x_obs_list) == 5 and len(opt._y_obs_list) == 5
        assert opt.ask(num_arms=2).shape == (2, 2)


def test_trailing_obs_includes_incumbent():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(123)
    for mode, cfg in [
        (TurboMode.TURBO_ONE, TurboOneConfig(trailing_obs=5)),
        (TurboMode.TURBO_ENN, TurboENNConfig(trailing_obs=5)),
    ]:
        opt = TurboOptimizer(bounds=bounds, mode=mode, rng=rng, config=cfg)
        for i in range(15):
            x = opt.ask(num_arms=2)
            y = (
                np.array([10.0, 9.0])
                if i == 0
                else np.array([5.0 - i * 0.1, 4.0 - i * 0.1])
            )
            opt.tell(x, y)
        assert opt.tr_obs_count <= 5
        assert opt.ask(num_arms=2).shape == (2, 2)


def test_turbo_optimizer_tell_without_yvar():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = TurboOptimizer(
        bounds=bounds, mode=TurboMode.TURBO_ENN, rng=np.random.default_rng(42)
    )
    for _ in range(2):
        x = opt.ask(num_arms=4)
        opt.tell(x, -np.sum(x**2, axis=1))
    x2 = opt.ask(num_arms=4)
    assert x2.shape == (4, 2) and np.all(x2 >= 0.0) and np.all(x2 <= 1.0)


def test_turbo_optimizer_yvar_policy_enforced():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    # Once yvar provided, must always provide
    opt = TurboOptimizer(
        bounds=bounds, mode=TurboMode.TURBO_ONE, rng=np.random.default_rng(0)
    )
    x0 = opt.ask(num_arms=2)
    y0 = -np.sum(x0**2, axis=1)
    opt.tell(x0, y0, 0.1 * np.ones_like(y0))
    x1 = opt.ask(num_arms=2)
    with pytest.raises(ValueError, match="y_var must be provided"):
        opt.tell(x1, -np.sum(x1**2, axis=1))
    # Once yvar omitted, must always omit
    opt2 = TurboOptimizer(
        bounds=bounds, mode=TurboMode.TURBO_ONE, rng=np.random.default_rng(0)
    )
    x0 = opt2.ask(num_arms=2)
    opt2.tell(x0, -np.sum(x0**2, axis=1))
    x1 = opt2.ask(num_arms=2)
    with pytest.raises(ValueError, match="y_var must be omitted"):
        opt2.tell(x1, -np.sum(x1**2, axis=1), 0.1 * np.ones(2))


def test_turbo_one_trust_region_update_is_noise_robust_to_spikes():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ONE,
        rng=np.random.default_rng(0),
        config=TurboOneConfig(num_init=1, num_candidates=16),
    )
    opt.ask(num_arms=1)
    opt.tell(np.zeros((1, 2)), np.array([0.0]), y_var=np.array([1e6]))
    opt.ask(num_arms=1)
    opt.tell(np.ones((1, 2)), np.array([100.0]), y_var=np.array([1e6]))
    assert opt._tr_state.best_value < 100.0
    y_tr_array = np.asarray(opt._y_tr_list, dtype=float)
    if y_tr_array.ndim == 2:
        y_tr_array = y_tr_array[:, 0]
    assert opt._tr_state.best_value == float(np.max(y_tr_array))


def test_turbo_enn_tr_values_use_posterior_mean_over_all_obs():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ENN,
        rng=np.random.default_rng(0),
        config=TurboENNConfig(num_init=1, num_candidates=16, acq_type="pareto", k=3),
    )
    opt.ask(num_arms=1)
    opt.tell(np.zeros((1, 2)), np.array([0.0]), y_var=np.array([1e6]))
    opt.ask(num_arms=1)
    opt.tell(np.ones((1, 2)), np.array([100.0]), y_var=np.array([1e6]))
    tr_vals = np.asarray(opt._y_tr_list, dtype=float)
    if tr_vals.ndim == 2:
        tr_vals = tr_vals[:, 0]
    assert tr_vals.shape == (2,)
    assert not np.allclose(tr_vals, np.array([0.0, 100.0]))
    assert np.all(tr_vals > 1.0) and np.all(tr_vals < 99.0)


def test_turbo_optimizer_no_trust_region_bounds_are_full_box():
    bounds = np.array([[-2.0, 2.0], [-1.0, 1.0], [0.0, 3.0]], dtype=float)
    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ZERO,
        rng=np.random.default_rng(0),
        config=TurboZeroConfig(tr_type="none", num_init=1, num_candidates=16),
    )
    x0 = opt.ask(num_arms=1)
    opt.tell(x0, np.array([1.0]))
    x_center = np.array([0.25, 0.5, 0.75], dtype=float)
    lb, ub = opt._tr_state.compute_bounds_1d(x_center)
    assert np.allclose(lb, 0.0) and np.allclose(ub, 1.0)


def test_turbo_optimizer_morbo_multi_objective():
    num_dim, num_metrics = 3, 2
    bounds = np.array([[0.0, 1.0]] * num_dim, dtype=float)
    rng = np.random.default_rng(42)
    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ENN,
        rng=rng,
        config=TurboENNConfig(tr_type="morbo", num_metrics=num_metrics, num_init=4),
    )
    for _ in range(3):
        x = opt.ask(num_arms=2)
        assert x.shape == (2, num_dim)
        y = rng.uniform(0.0, 1.0, size=(2, num_metrics))
        y_est = opt.tell(x, y)
        assert y_est.shape == (2, num_metrics)
    x = opt.ask(num_arms=2)
    assert x.shape == (2, num_dim)
    y_est = opt.tell(x, rng.uniform(0.0, 1.0, size=(2, num_metrics)))
    assert y_est.shape == (2, num_metrics)
    tr_len = opt.tr_length
    assert tr_len is not None and 0.0 < tr_len <= 1.0
