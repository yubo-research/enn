from __future__ import annotations

import conftest
import numpy as np
import pytest

from enn.turbo.config import (
    AcqType,
    CandidateGenConfig,
    CandidateRV,
    ENNFitConfig,
    ENNSurrogateConfig,
    MorboTRConfig,
    MultiObjectiveConfig,
    NoTRConfig,
    OptimizerConfig,
    TurboTRConfig,
    turbo_enn_config,
    turbo_one_config,
    turbo_zero_config,
)
from enn.turbo.python_fallback.optimizer import (
    create_optimizer as create_fallback_optimizer,
)


def _make_fallback_optimizer(*, bounds, config, rng):
    return create_fallback_optimizer(bounds=bounds, config=config, rng=rng)


def _make_rust_optimizer(*, bounds, config, rng):
    from enn import create_optimizer

    return create_optimizer(bounds=bounds, config=config, rng=rng)


def test_turbo_fallback_called_during_init_with_observations():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    config = turbo_zero_config(num_init=10)
    opt = _make_rust_optimizer(bounds=bounds, config=config, rng=rng)
    x1 = opt.ask(num_arms=2)
    y1 = conftest.sphere_objective(x1)
    opt.tell(x1, y1)
    x2 = opt.ask(num_arms=2)
    assert x2.shape == (2, 2)
    init = opt.init_progress
    assert init is not None
    init_idx, num_init = init
    assert init_idx <= num_init


def _run_bo(config: OptimizerConfig, num_steps: int = 15) -> float:
    from enn import create_optimizer

    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    best = -np.inf
    for _ in range(num_steps):
        x = opt.ask(num_arms=4)
        y = conftest.sphere_objective(x)
        opt.tell(x, y)
        best = max(best, float(np.max(y)))
    return best


def test_turbo_zero_ask_tell_and_shape():
    from enn import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = create_optimizer(bounds=bounds, config=turbo_zero_config(), rng=rng)
    x0 = opt.ask(num_arms=4)
    assert x0.shape == (4, 2) and np.all(x0 >= 0.0) and np.all(x0 <= 1.0)
    opt.tell(x0, conftest.sphere_objective(x0))
    x1 = opt.ask(num_arms=4)
    assert x1.shape == (4, 2)


def test_optimizer_accepts_list_bounds():
    from enn import create_optimizer

    opt = create_optimizer(
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        config=turbo_zero_config(),
        rng=np.random.default_rng(0),
    )
    assert opt.ask(num_arms=2).shape == (2, 2)


def test_optimizer_uniform_candidates_never_calls_sobol():
    from unittest import mock

    from enn import create_optimizer
    from enn.turbo.optimizer_config import turbo_zero_config

    def _sobol_raises(*args, **kwargs):
        raise RuntimeError(
            "Sobol should not be constructed for candidate_rv=CandidateRV.UNIFORM"
        )

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    with mock.patch("scipy.stats.qmc.Sobol", side_effect=_sobol_raises):
        opt = create_optimizer(
            bounds=bounds,
            config=turbo_zero_config(candidate_rv=CandidateRV.UNIFORM),
            rng=rng,
        )
        x0 = opt.ask(num_arms=4)
    assert x0.shape == (4, 2)


def test_optimizer_accepts_base_config_as_turbo_zero():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = _make_rust_optimizer(
        bounds=bounds,
        config=OptimizerConfig(),
        rng=np.random.default_rng(0),
    )
    x = opt.ask(num_arms=2)
    assert x.shape == (2, 2)


def test_turbo_one_improves_on_sphere():
    assert _run_bo(turbo_one_config(), num_steps=12) > -0.5


def test_turbo_one_pareto_ask_tell_runs():
    from enn import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    config = turbo_one_config(acq_type=AcqType.PARETO, num_init=2)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    for _ in range(3):
        x = opt.ask(num_arms=2)
        assert x.shape == (2, 2)
        y = conftest.sphere_objective(x)
        opt.tell(x, y)
    x_final = opt.ask(num_arms=2)
    assert x_final.shape == (2, 2)


def test_turbo_one_with_y_var_uses_noisy_gp():
    from enn import create_optimizer
    from enn.turbo.python_fallback.turbo_gp_noisy import TurboGPNoisy

    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=turbo_one_config(), rng=rng)
    for _ in range(5):
        x = opt.ask(num_arms=4)
        y = conftest.sphere_objective(x)
        opt.tell(x, y, rng.uniform(0.01, 0.1, size=y.shape))
    x = opt.ask(num_arms=4)
    assert x.shape == (4, 2)
    assert isinstance(opt._surrogate._model, TurboGPNoisy)


def test_turbo_zero_reasonable_on_sphere():
    assert _run_bo(turbo_zero_config(), num_steps=12) > -1.5


def test_turbo_enn_uses_enn_and_is_reasonable():
    assert _run_bo(turbo_enn_config(), num_steps=12) > -1.5


def test_turbo_enn_with_k_none_fits_hyperparameters():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = _make_rust_optimizer(bounds=bounds, config=turbo_enn_config(), rng=rng)
    x0 = opt.ask(num_arms=4)
    assert x0.shape == (4, 2) and np.all(x0 >= 0.0) and np.all(x0 <= 1.0)
    opt.tell(x0, -np.sum(x0**2, axis=1))
    x1 = opt.ask(num_arms=4)
    assert x1.shape == (4, 2) and np.all(x1 >= 0.0) and np.all(x1 <= 1.0)


def test_turbo_enn_config_scale_x_flag_runs():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    from enn.turbo.optimizer_config import ENNSurrogateConfig

    opt = _make_rust_optimizer(
        bounds=bounds,
        config=turbo_enn_config(enn=ENNSurrogateConfig(scale_x=True)),
        rng=np.random.default_rng(0),
    )
    num_arms = 3
    for _ in range(2):
        x = opt.ask(num_arms=num_arms)
        opt.tell(x, -np.sum(x**2, axis=1))
    assert opt.ask(num_arms=num_arms).shape == (num_arms, 2)


def test_find_x_center_uses_top_k_for_mu_single_objective():
    from enn.turbo.python_fallback.components.posterior_result import PosteriorResult
    from enn.turbo.optimizer_config import TurboTRConfig

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = _make_fallback_optimizer(
        bounds=bounds,
        config=turbo_one_config(
            trust_region=TurboTRConfig(noise_aware=True),
        ),
        rng=rng,
    )
    x_obs = rng.uniform(0.0, 1.0, size=(10, 2))
    y_obs = np.arange(10, dtype=float)
    seen: dict[str, tuple[int, int]] = {}

    def _predict(x):
        seen["shape"] = x.shape
        mu = np.arange(x.shape[0], dtype=float).reshape(-1, 1)
        return PosteriorResult(mu=mu, sigma=None)

    opt._surrogate.predict = _predict
    for x, y in zip(x_obs, y_obs):
        opt._x_obs.append(x)
        opt._y_obs.append(np.array([y]))
    opt._update_incumbent()
    center = opt._find_x_center(x_obs, y_obs)
    assert center.shape == (2,)
    assert seen["shape"] == (10, 2)


def test_find_x_center_uses_top_k_union_for_multiobjective():
    from enn.turbo.python_fallback.components.posterior_result import PosteriorResult
    from enn.turbo.optimizer_config import MorboTRConfig

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = _make_fallback_optimizer(
        bounds=bounds,
        config=turbo_one_config(
            trust_region=MorboTRConfig(
                noise_aware=True,
                multi_objective=MultiObjectiveConfig(num_metrics=2),
            ),
        ),
        rng=rng,
    )
    x_obs = rng.uniform(0.0, 1.0, size=(5, 2))
    y_obs = np.array(
        [
            [10.0, 0.0],
            [9.0, 1.0],
            [0.0, 10.0],
            [1.0, 9.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    seen: dict[str, tuple[int, int]] = {}

    def _predict(x):
        seen["shape"] = x.shape
        mu = np.zeros((x.shape[0], 2), dtype=float)
        return PosteriorResult(mu=mu, sigma=None)

    opt._surrogate.predict = _predict
    for x, y in zip(x_obs, y_obs):
        opt._x_obs.append(x)
        opt._y_obs.append(y)
    opt._update_incumbent()
    center = opt._find_x_center(x_obs, y_obs)
    assert center.shape == (2,)
    assert seen["shape"] == (5, 2)


def test_update_incumbent_uses_tracker_not_surrogate_candidate_api():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = _make_fallback_optimizer(
        bounds=bounds,
        config=turbo_one_config(),
        rng=rng,
    )
    ask_calls: list[bool] = []
    real_ask = opt._incumbent_tracker.ask

    def _spy_ask():
        ask_calls.append(True)
        return real_ask()

    opt._incumbent_tracker.ask = _spy_ask  # type: ignore[method-assign]
    x = rng.uniform(0.0, 1.0, size=(4, 2))
    y = np.array([0.1, 0.5, 0.2, 0.9], dtype=float)
    opt.tell(x, y.reshape(-1, 1))
    assert ask_calls
    assert opt._incumbent_tracker.observation_count() == len(opt._y_obs)
    assert opt._incumbent_idx is not None


def test_optimizer_keeps_all_observations():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    cases = [
        (turbo_one_config(), _make_fallback_optimizer),
        (turbo_enn_config(), _make_rust_optimizer),
    ]
    for cfg, make in cases:
        opt = make(bounds=bounds, config=cfg, rng=rng)
        for _ in range(10):
            x = opt.ask(num_arms=2)
            opt.tell(x, -np.sum(x**2, axis=1))
        assert opt._x_obs.view().shape[0] == 20 and opt._y_obs.view().shape[0] == 20
        assert opt.ask(num_arms=2).shape == (2, 2)


def test_incumbent_preserved_across_many_tells():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(123)
    cases = [
        (turbo_one_config(), _make_fallback_optimizer),
        (turbo_enn_config(), _make_rust_optimizer),
    ]
    for cfg, make in cases:
        opt = make(bounds=bounds, config=cfg, rng=rng)
        for i in range(15):
            x = opt.ask(num_arms=2)
            y = (
                np.array([10.0, 9.0])
                if i == 0
                else np.array([5.0 - i * 0.1, 4.0 - i * 0.1])
            )
            opt.tell(x, y)
        assert opt.tr_obs_count == 30
        assert opt.ask(num_arms=2).shape == (2, 2)


def test_optimizer_tell_without_yvar():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = _make_rust_optimizer(
        bounds=bounds, config=turbo_enn_config(), rng=np.random.default_rng(42)
    )
    for _ in range(2):
        x = opt.ask(num_arms=4)
        opt.tell(x, -np.sum(x**2, axis=1))
    x2 = opt.ask(num_arms=4)
    assert x2.shape == (4, 2) and np.all(x2 >= 0.0) and np.all(x2 <= 1.0)


def test_optimizer_yvar_policy_enforced():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = _make_fallback_optimizer(
        bounds=bounds, config=turbo_one_config(), rng=np.random.default_rng(0)
    )
    x0 = opt.ask(num_arms=2)
    y0 = -np.sum(x0**2, axis=1)
    opt.tell(x0, y0, 0.1 * np.ones_like(y0))
    x1 = opt.ask(num_arms=2)
    with pytest.raises(ValueError, match="y_var must be provided"):
        opt.tell(x1, -np.sum(x1**2, axis=1))
    opt2 = _make_fallback_optimizer(
        bounds=bounds, config=turbo_one_config(), rng=np.random.default_rng(0)
    )
    x0 = opt2.ask(num_arms=2)
    opt2.tell(x0, -np.sum(x0**2, axis=1))
    x1 = opt2.ask(num_arms=2)
    with pytest.raises(ValueError, match="y_var must be omitted"):
        opt2.tell(x1, -np.sum(x1**2, axis=1), 0.1 * np.ones(2))


def test_turbo_one_trust_region_update_is_noise_robust_to_spikes():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = _make_fallback_optimizer(
        bounds=bounds,
        config=turbo_one_config(
            num_init=1,
            num_candidates=16,
            trust_region=TurboTRConfig(noise_aware=True),
        ),
        rng=np.random.default_rng(0),
    )
    opt.ask(num_arms=1)
    opt.tell(np.zeros((1, 2)), np.array([0.0]), y_var=np.array([1e6]))
    opt.ask(num_arms=1)
    opt.tell(np.ones((1, 2)), np.array([100.0]), y_var=np.array([1e6]))
    assert opt._tr_state.best_value < 100.0


def test_turbo_enn_tr_values_do_not_require_full_history_denoising():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    from enn.turbo.config import (
        AcqType,
        CandidateGenConfig,
        ENNSurrogateConfig,
    )

    opt = _make_rust_optimizer(
        bounds=bounds,
        config=turbo_enn_config(
            enn=ENNSurrogateConfig(k=3),
            candidates=CandidateGenConfig(num_candidates=16),
            num_init=1,
            acq_type=AcqType.PARETO,
        ),
        rng=np.random.default_rng(0),
    )
    opt.ask(num_arms=1)
    opt.tell(np.zeros((1, 2)), np.array([0.0]), y_var=np.array([1e6]))
    opt.ask(num_arms=1)
    opt.tell(np.ones((1, 2)), np.array([100.0]), y_var=np.array([1e6]))
    assert opt.ask(num_arms=1).shape == (1, 2)


def test_optimizer_no_trust_region_bounds_are_full_box():
    bounds = np.array([[-2.0, 2.0], [-1.0, 1.0], [0.0, 3.0]], dtype=float)
    opt = _make_rust_optimizer(
        bounds=bounds,
        config=turbo_zero_config(
            trust_region=NoTRConfig(), num_init=1, num_candidates=16
        ),
        rng=np.random.default_rng(0),
    )
    x0 = opt.ask(num_arms=1)
    opt.tell(x0, np.array([1.0]))
    x1 = opt.ask(num_arms=1)
    assert x1.shape == (1, 3)
    assert np.all(x1 >= bounds[:, 0] - 1e-9)
    assert np.all(x1 <= bounds[:, 1] + 1e-9)


def test_optimizer_morbo_multi_objective():
    num_dim, num_metrics = 2, 2
    bounds = np.array([[0.0, 1.0]] * num_dim, dtype=float)
    rng = np.random.default_rng(42)
    opt = _make_rust_optimizer(
        bounds=bounds,
        config=turbo_enn_config(
            enn=ENNSurrogateConfig(
                k=2,
                fit=ENNFitConfig(num_fit_samples=1, num_fit_candidates=1),
            ),
            trust_region=MorboTRConfig(
                multi_objective=MultiObjectiveConfig(num_metrics=num_metrics)
            ),
            num_init=1,
            candidates=CandidateGenConfig(num_candidates=2),
            acq_type=AcqType.THOMPSON,
        ),
        rng=rng,
    )
    x = opt.ask(num_arms=1)
    assert x.shape == (1, num_dim)
    y = rng.uniform(0.0, 1.0, size=(1, num_metrics))
    y_est = opt.tell(x, y)
    assert y_est.shape == (1, num_metrics)
    tr_len = opt.tr_length
    assert tr_len is not None and 0.0 < tr_len <= 1.0
