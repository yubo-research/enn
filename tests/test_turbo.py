from __future__ import annotations

import conftest
import pytest

from enn.turbo.turbo_mode import TurboMode
from enn.turbo.turbo_utils import to_unit


def _run_bo(mode: TurboMode, num_steps: int = 15) -> float:
    import numpy as np

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
    import numpy as np

    from enn import Turbo

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = Turbo(
        bounds=bounds,
        mode=TurboMode.TURBO_ZERO,
        rng=rng,
    )
    x0 = opt.ask(num_arms=4)
    assert x0.shape == (4, 2)
    assert np.all(x0 >= 0.0) and np.all(x0 <= 1.0)
    y0 = conftest.sphere_objective(x0)
    opt.tell(x0, y0)
    x1 = opt.ask(num_arms=4)
    assert x1.shape == (4, 2)


def test_turbo_optimizer_accepts_list_bounds():
    import numpy as np

    from enn import Turbo

    bounds = [[0.0, 1.0], [0.0, 1.0]]
    rng = np.random.default_rng(0)
    opt = Turbo(bounds=bounds, mode=TurboMode.TURBO_ZERO, rng=rng)
    x0 = opt.ask(num_arms=2)
    assert x0.shape == (2, 2)


def test_turbo_optimizer_requires_mode_specific_config():
    import numpy as np

    from enn.turbo.turbo_config import TurboConfig
    from enn.turbo.turbo_optimizer import TurboOptimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="requires TurboENNConfig"):
        TurboOptimizer(
            bounds=bounds,
            mode=TurboMode.TURBO_ENN,
            rng=rng,
            config=TurboConfig(),
        )


def test_turbo_one_improves_on_sphere():
    best = _run_bo(TurboMode.TURBO_ONE, num_steps=12)
    assert best > -0.5


def test_turbo_one_with_y_var_uses_noisy_gp():
    import numpy as np

    from enn import Turbo
    from enn.turbo.turbo_gp_noisy import TurboGPNoisy

    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = Turbo(bounds=bounds, mode=TurboMode.TURBO_ONE, rng=rng)

    for _ in range(5):
        x = opt.ask(num_arms=4)
        y = conftest.sphere_objective(x)
        y_var = rng.uniform(0.01, 0.1, size=y.shape)
        opt.tell(x, y, y_var)

    x = opt.ask(num_arms=4)
    assert x.shape == (4, 2)

    gp_model = opt._mode_impl._gp_model
    assert gp_model is not None
    assert isinstance(gp_model, TurboGPNoisy)


def test_turbo_zero_reasonable_on_sphere():
    best = _run_bo(TurboMode.TURBO_ZERO, num_steps=12)
    assert best > -1.5


def test_turbo_enn_uses_enn_and_is_reasonable():
    best = _run_bo(TurboMode.TURBO_ENN, num_steps=12)
    assert best > -1.5


def test_turbo_enn_with_k_none_fits_hyperparameters():
    import numpy as np

    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_optimizer import TurboOptimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    from enn.turbo.turbo_config import TurboENNConfig

    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ENN,
        rng=rng,
        config=TurboENNConfig(k=None),
    )
    x0 = opt.ask(num_arms=4)
    assert x0.shape == (4, 2)
    assert np.all(x0 >= 0.0) and np.all(x0 <= 1.0)
    y0 = -np.sum(x0**2, axis=1)
    opt.tell(x0, y0)
    x1 = opt.ask(num_arms=4)
    assert x1.shape == (4, 2)
    assert np.all(x1 >= 0.0) and np.all(x1 <= 1.0)


def test_turbo_enn_config_scale_x_flag_runs():
    import numpy as np

    from enn.turbo.turbo_config import TurboENNConfig
    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_optimizer import TurboOptimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ENN,
        rng=rng,
        config=TurboENNConfig(scale_x=True),
    )
    x0 = opt.ask(num_arms=3)
    y0 = -np.sum(x0**2, axis=1)
    opt.tell(x0, y0)
    x1 = opt.ask(num_arms=3)
    assert x1.shape == (3, 2)


def test_turbo_optimizer_with_trailing_obs():
    import numpy as np

    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_optimizer import TurboOptimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)

    from enn.turbo.turbo_config import TurboENNConfig, TurboOneConfig

    for mode, cfg in [
        (TurboMode.TURBO_ONE, TurboOneConfig(trailing_obs=5)),
        (TurboMode.TURBO_ENN, TurboENNConfig(trailing_obs=5)),
    ]:
        opt = TurboOptimizer(
            bounds=bounds,
            mode=mode,
            rng=rng,
            config=cfg,
        )
        for i in range(10):
            x = opt.ask(num_arms=2)
            assert x.shape == (2, 2)
            y = -np.sum(x**2, axis=1)
            opt.tell(x, y)
        assert len(opt._x_obs_list) == 5
        assert len(opt._y_obs_list) == 5
        x_final = opt.ask(num_arms=2)
        assert x_final.shape == (2, 2)


def test_trailing_obs_includes_incumbent():
    import numpy as np

    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_optimizer import TurboOptimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(123)

    from enn.turbo.turbo_config import TurboENNConfig, TurboOneConfig

    for mode, cfg in [
        (TurboMode.TURBO_ONE, TurboOneConfig(trailing_obs=5)),
        (TurboMode.TURBO_ENN, TurboENNConfig(trailing_obs=5)),
    ]:
        opt = TurboOptimizer(
            bounds=bounds,
            mode=mode,
            rng=rng,
            config=cfg,
        )
        for i in range(15):
            x = opt.ask(num_arms=2)
            if i == 0:
                y = np.array([10.0, 9.0])
            else:
                y = np.array([5.0 - i * 0.1, 4.0 - i * 0.1])
            opt.tell(x, y)

        assert opt.tr_obs_count <= 5

        x_new = opt.ask(num_arms=2)
        assert x_new.shape == (2, 2)
        assert opt.tr_obs_count <= 5


def test_latin_hypercube_stratification_and_bounds():
    import numpy as np

    from enn.turbo.turbo_utils import latin_hypercube

    rng = np.random.default_rng(0)
    n = 8
    d = 3
    x = latin_hypercube(n, d, rng=rng)
    assert x.shape == (n, d)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    for j in range(d):
        xs = np.sort(x[:, j])
        for k in range(n):
            lo = k / n
            hi = (k + 1) / n
            in_bin = (xs >= lo) & (xs <= hi + 1e-8)
            assert np.any(in_bin)


def test_argmax_random_tie_uses_rng_and_is_deterministic():
    import numpy as np

    from enn.turbo.turbo_utils import argmax_random_tie

    values = np.array([1.0, 2.0, 2.0, 0.0], dtype=float)
    rng = np.random.default_rng(0)
    idx1 = argmax_random_tie(values, rng=rng)
    assert idx1 in (1, 2)
    rng = np.random.default_rng(0)
    idx2 = argmax_random_tie(values, rng=rng)
    assert idx1 == idx2


def test_trust_region_state_update_and_restart_and_bounds():
    import numpy as np

    from enn.turbo.turbo_trust_region import TurboTrustRegion

    state = TurboTrustRegion(num_dim=2, num_arms=2)
    values = []
    for v in [0.0, 1.0, 2.0]:
        values.append(v)
        state.update(np.array(values, dtype=float))
    x_center = np.zeros(2, dtype=float)
    lb, ub = state.compute_bounds_1d(x_center)
    assert lb.shape == (2,)
    assert ub.shape == (2,)
    state.length = state.length_min / 2.0
    assert state.needs_restart()
    state.restart()
    assert state.length == state.length_init


@pytest.mark.parametrize(
    "mode", [TurboMode.TURBO_ZERO, TurboMode.TURBO_ONE, TurboMode.TURBO_ENN]
)
def test_turbo_behavior_independent_of_affine_x(mode: TurboMode) -> None:
    import numpy as np

    from enn import Turbo

    bounds1 = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    bounds2 = np.array([[2.0, 4.0], [-3.0, 1.0]], dtype=float)
    num_arms = 4
    num_steps = 8
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    opt1 = Turbo(
        bounds=bounds1,
        mode=mode,
        rng=rng1,
    )
    opt2 = Turbo(
        bounds=bounds2,
        mode=mode,
        rng=rng2,
    )

    for _ in range(num_steps):
        x1 = opt1.ask(num_arms=num_arms)
        x2 = opt2.ask(num_arms=num_arms)
        u1 = to_unit(x1, bounds1)
        u2 = to_unit(x2, bounds2)
        assert np.allclose(u1, u2)
        z1 = 2.0 * u1 - 1.0
        z2 = 2.0 * u2 - 1.0
        y1 = conftest.sphere_objective(z1)
        y2 = conftest.sphere_objective(z2)
        assert np.allclose(y1, y2)
        opt1.tell(x1, y1)
        opt2.tell(x2, y2)


@pytest.mark.parametrize("mode", [TurboMode.TURBO_ZERO, TurboMode.TURBO_ENN])
def test_turbo_behavior_independent_of_affine_y(mode: TurboMode) -> None:
    import numpy as np

    from enn import Turbo

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 4
    num_steps = 8

    def run_with_transform(scale: float, shift: float) -> np.ndarray:
        rng = np.random.default_rng(0)
        opt = Turbo(
            bounds=bounds,
            mode=mode,
            rng=rng,
        )
        unit_trajectory = []
        for _ in range(num_steps):
            x = opt.ask(num_arms=num_arms)
            # With these bounds, x already lives in unit space.
            u = x.copy()
            z = 2.0 * u - 1.0
            base_y = conftest.sphere_objective(z)
            y = scale * base_y + shift
            opt.tell(x, y)
            unit_trajectory.append(u)
        return np.stack(unit_trajectory, axis=0)

    traj_base = run_with_transform(scale=1.0, shift=0.0)
    traj_affine = run_with_transform(scale=2.0, shift=0.5)
    # The sequence of unit-space query points should be invariant to affine
    # rescalings (scale and center) of the observed y values.
    # Note: TURBO_ONE is excluded because it uses a relative threshold
    # (1e-3 * abs(best_value)) for improvement detection, which breaks affine
    # invariance by design (matching the reference implementation).
    assert np.allclose(traj_base, traj_affine)


def test_sobol_perturb_np_shape_and_bounds():
    import numpy as np
    from scipy.stats import qmc

    from enn.turbo.turbo_utils import sobol_perturb_np

    num_candidates = 10
    num_dim = 3
    x_center = np.array([0.5, 0.5, 0.5], dtype=float)
    lb = np.array([0.0, 0.0, 0.0], dtype=float)
    ub = np.array([1.0, 1.0, 1.0], dtype=float)
    mask = np.ones((num_candidates, num_dim), dtype=bool)
    sobol_engine = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    candidates = sobol_perturb_np(
        x_center, lb, ub, num_candidates, mask, sobol_engine=sobol_engine
    )
    assert candidates.shape == (num_candidates, num_dim)
    assert np.all(candidates >= lb)
    assert np.all(candidates <= ub)


def test_sobol_perturb_np_mask_application():
    import numpy as np
    from scipy.stats import qmc

    from enn.turbo.turbo_utils import sobol_perturb_np

    num_candidates = 5
    num_dim = 3
    x_center = np.array([0.5, 0.5, 0.5], dtype=float)
    lb = np.array([0.0, 0.0, 0.0], dtype=float)
    ub = np.array([1.0, 1.0, 1.0], dtype=float)
    mask = np.zeros((num_candidates, num_dim), dtype=bool)
    mask[:, 0] = True
    mask[0, 1] = True
    sobol_engine = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    candidates = sobol_perturb_np(
        x_center, lb, ub, num_candidates, mask, sobol_engine=sobol_engine
    )
    assert candidates.shape == (num_candidates, num_dim)
    for i in range(num_candidates):
        for j in range(num_dim):
            if mask[i, j]:
                assert candidates[i, j] != x_center[j]
                assert lb[j] <= candidates[i, j] <= ub[j]
            else:
                assert candidates[i, j] == x_center[j]


def test_sobol_perturb_np_deterministic():
    import numpy as np
    from scipy.stats import qmc

    from enn.turbo.turbo_utils import sobol_perturb_np

    num_candidates = 8
    num_dim = 2
    x_center = np.array([0.5, 0.5], dtype=float)
    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([1.0, 1.0], dtype=float)
    mask = np.ones((num_candidates, num_dim), dtype=bool)
    sobol_engine1 = qmc.Sobol(d=num_dim, scramble=True, seed=42)
    sobol_engine2 = qmc.Sobol(d=num_dim, scramble=True, seed=42)
    candidates1 = sobol_perturb_np(
        x_center, lb, ub, num_candidates, mask, sobol_engine=sobol_engine1
    )
    candidates2 = sobol_perturb_np(
        x_center, lb, ub, num_candidates, mask, sobol_engine=sobol_engine2
    )
    assert np.allclose(candidates1, candidates2)


def test_raasp_shape_and_bounds():
    import numpy as np
    from scipy.stats import qmc

    from enn.turbo.turbo_utils import raasp

    num_candidates = 10
    num_dim = 3
    x_center = np.array([0.5, 0.5, 0.5], dtype=float)
    lb = np.array([0.0, 0.0, 0.0], dtype=float)
    ub = np.array([1.0, 1.0, 1.0], dtype=float)
    rng = np.random.default_rng(0)
    sobol_engine = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    candidates = raasp(
        x_center,
        lb,
        ub,
        num_candidates,
        num_pert=20,
        rng=rng,
        sobol_engine=sobol_engine,
    )
    assert candidates.shape == (num_candidates, num_dim)
    assert np.all(candidates >= lb)
    assert np.all(candidates <= ub)


def test_raasp_at_least_one_dimension_perturbed():
    import numpy as np
    from scipy.stats import qmc

    from enn.turbo.turbo_utils import raasp

    num_candidates = 20
    num_dim = 5
    x_center = np.array([0.5] * num_dim, dtype=float)
    lb = np.array([0.0] * num_dim, dtype=float)
    ub = np.array([1.0] * num_dim, dtype=float)
    rng = np.random.default_rng(0)
    sobol_engine = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    candidates = raasp(
        x_center,
        lb,
        ub,
        num_candidates,
        num_pert=20,
        rng=rng,
        sobol_engine=sobol_engine,
    )
    for i in range(num_candidates):
        diff = np.abs(candidates[i] - x_center)
        assert np.any(diff > 1e-10)


def test_raasp_deterministic():
    import numpy as np
    from scipy.stats import qmc

    from enn.turbo.turbo_utils import raasp

    num_candidates = 8
    num_dim = 2
    x_center = np.array([0.5, 0.5], dtype=float)
    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([1.0, 1.0], dtype=float)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    sobol_engine1 = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    sobol_engine2 = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    candidates1 = raasp(
        x_center,
        lb,
        ub,
        num_candidates,
        num_pert=20,
        rng=rng1,
        sobol_engine=sobol_engine1,
    )
    candidates2 = raasp(
        x_center,
        lb,
        ub,
        num_candidates,
        num_pert=20,
        rng=rng2,
        sobol_engine=sobol_engine2,
    )
    assert np.allclose(candidates1, candidates2)


def test_raasp_probability_scaling():
    import numpy as np
    from scipy.stats import qmc

    from enn.turbo.turbo_utils import raasp

    num_candidates = 100
    num_dim_low = 5
    num_dim_high = 100
    x_center_low = np.array([0.5] * num_dim_low, dtype=float)
    x_center_high = np.array([0.5] * num_dim_high, dtype=float)
    lb_low = np.array([0.0] * num_dim_low, dtype=float)
    ub_low = np.array([1.0] * num_dim_low, dtype=float)
    lb_high = np.array([0.0] * num_dim_high, dtype=float)
    ub_high = np.array([1.0] * num_dim_high, dtype=float)
    rng = np.random.default_rng(0)
    sobol_engine_low = qmc.Sobol(d=num_dim_low, scramble=True, seed=0)
    sobol_engine_high = qmc.Sobol(d=num_dim_high, scramble=True, seed=0)
    candidates_low = raasp(
        x_center_low,
        lb_low,
        ub_low,
        num_candidates,
        num_pert=20,
        rng=rng,
        sobol_engine=sobol_engine_low,
    )
    rng = np.random.default_rng(0)
    candidates_high = raasp(
        x_center_high,
        lb_high,
        ub_high,
        num_candidates,
        num_pert=20,
        rng=rng,
        sobol_engine=sobol_engine_high,
    )
    diff_low = np.sum(np.abs(candidates_low - x_center_low) > 1e-10, axis=1)
    diff_high = np.sum(np.abs(candidates_high - x_center_high) > 1e-10, axis=1)
    mean_perturbed_low = np.mean(diff_low) / num_dim_low
    mean_perturbed_high = np.mean(diff_high) / num_dim_high
    assert mean_perturbed_low > mean_perturbed_high


def test_to_unit_and_from_unit_roundtrip():
    import numpy as np

    from enn.turbo.turbo_utils import from_unit

    bounds = np.array([[0.0, 2.0], [-1.0, 1.0], [5.0, 10.0]], dtype=float)
    x_original = np.array([[1.0, 0.0, 7.5], [0.5, -0.5, 8.0]], dtype=float)
    x_unit = to_unit(x_original, bounds)
    assert x_unit.shape == x_original.shape
    assert np.all(x_unit >= 0.0) and np.all(x_unit <= 1.0)
    x_roundtrip = from_unit(x_unit, bounds)
    assert np.allclose(x_original, x_roundtrip)


def test_to_unit_bounds_validation():
    import numpy as np

    bounds_invalid = np.array([[1.0, 0.0]], dtype=float)
    x = np.array([[0.5]], dtype=float)
    with pytest.raises(ValueError):
        to_unit(x, bounds_invalid)


def test_select_uniform_shape_and_uniformity():
    import numpy as np

    from enn.turbo.proposal import select_uniform

    num_candidates = 128
    num_dim = 4
    num_arms = 8
    x_cand = np.random.default_rng(0).random((num_candidates, num_dim))
    bounds = np.array([[0.0, 1.0]] * num_dim, dtype=float)
    rng = np.random.default_rng(42)
    from_unit_fn = conftest.make_from_unit_fn(bounds)

    selected = select_uniform(x_cand, num_arms, num_dim, rng, from_unit_fn)
    assert selected.shape == (num_arms, num_dim)
    assert len(np.unique([tuple(row) for row in selected], axis=0)) == num_arms


def test_select_uniform_validation():
    import numpy as np

    from enn.turbo.proposal import select_uniform

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    from_unit_fn = conftest.make_from_unit_fn(bounds)

    x_cand_wrong_dim = np.random.default_rng(0).random((10, 3))
    with pytest.raises(ValueError):
        select_uniform(x_cand_wrong_dim, 5, 2, rng, from_unit_fn)

    x_cand_too_few = np.random.default_rng(0).random((3, 2))
    with pytest.raises(ValueError):
        select_uniform(x_cand_too_few, 5, 2, rng, from_unit_fn)


def test_select_gp_thompson_uses_gp_and_returns_correct_shape():
    import numpy as np

    from enn.turbo.proposal import select_gp_thompson

    num_candidates = 30
    num_dim = 2
    num_arms = 5
    x_cand = np.random.default_rng(0).random((num_candidates, num_dim))
    x_obs = np.random.default_rng(1).random((15, num_dim))
    y_obs = (x_obs.sum(axis=1)).tolist()
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    from_unit_fn = conftest.make_from_unit_fn(bounds)
    select_sobol_fn = conftest.make_select_sobol_fn(bounds, rng)

    selected, new_mean, new_std, _ = select_gp_thompson(
        x_cand,
        num_arms,
        x_obs.tolist(),
        y_obs,
        num_dim,
        gp_num_steps=20,
        rng=rng,
        gp_y_mean=0.0,
        gp_y_std=1.0,
        select_sobol_fn=select_sobol_fn,
        from_unit_fn=from_unit_fn,
    )
    assert selected.shape == (num_arms, num_dim)
    assert isinstance(new_mean, float)
    assert isinstance(new_std, float)
    assert new_std > 0.0
    assert np.all(selected >= bounds[:, 0]) and np.all(selected <= bounds[:, 1])


def test_select_gp_thompson_fallback_on_empty_observations():
    import numpy as np

    from enn.turbo.proposal import select_gp_thompson

    num_candidates = 20
    num_dim = 2
    num_arms = 3
    x_cand = np.random.default_rng(0).random((num_candidates, num_dim))
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    from_unit_fn = conftest.make_from_unit_fn(bounds)

    fallback_called = False

    def select_sobol_fn(x, n):
        nonlocal fallback_called
        fallback_called = True
        idx = rng.choice(x.shape[0], size=n, replace=False)
        return from_unit_fn(x[idx])

    selected, mean, std, _ = select_gp_thompson(
        x_cand,
        num_arms,
        [],
        [],
        num_dim,
        gp_num_steps=20,
        rng=rng,
        gp_y_mean=0.0,
        gp_y_std=1.0,
        select_sobol_fn=select_sobol_fn,
        from_unit_fn=from_unit_fn,
    )
    assert fallback_called
    assert selected.shape == (num_arms, num_dim)
    assert mean == 0.0
    assert std == 1.0


def test_fit_gp_returns_model_with_valid_data():
    import numpy as np

    from enn.turbo.turbo_utils import fit_gp

    num_obs = 20
    num_dim = 3
    x_obs = np.random.default_rng(0).random((num_obs, num_dim))
    y_obs = (
        x_obs.sum(axis=1) + 0.1 * np.random.default_rng(1).standard_normal(num_obs)
    ).tolist()
    model, likelihood, y_mean, y_std = fit_gp(
        x_obs.tolist(), y_obs, num_dim, num_steps=10
    )
    assert model is not None
    assert likelihood is not None
    assert isinstance(y_mean, float)
    assert isinstance(y_std, float)
    assert y_std > 0.0


def test_fit_gp_returns_none_with_empty_data_and_returns_model_with_single_obs():
    import numpy as np

    from enn.turbo.turbo_utils import fit_gp

    num_dim = 2
    model_empty, likelihood_empty, mean_empty, std_empty = fit_gp(
        [], [], num_dim, num_steps=10
    )
    assert model_empty is None
    assert likelihood_empty is None
    assert mean_empty == 0.0
    assert std_empty == 1.0

    x_single = np.random.default_rng(0).random((1, num_dim))
    y_single = [1.0]
    model_single, likelihood_single, mean_single, std_single = fit_gp(
        x_single.tolist(), y_single, num_dim, num_steps=0
    )
    assert model_single is not None
    assert likelihood_single is not None
    assert isinstance(mean_single, float)
    assert mean_single == 1.0
    assert std_single == 1.0


def test_fit_gp_with_y_var_list_uses_noisy_model():
    import numpy as np

    from enn.turbo.turbo_gp_noisy import TurboGPNoisy
    from enn.turbo.turbo_utils import fit_gp

    num_obs = 20
    num_dim = 3
    rng = np.random.default_rng(42)
    x_obs = rng.random((num_obs, num_dim))
    y_obs = (x_obs.sum(axis=1) + 0.1 * rng.standard_normal(num_obs)).tolist()
    y_var = rng.uniform(0.01, 0.1, size=num_obs).tolist()

    model, likelihood, y_mean, y_std = fit_gp(
        x_obs.tolist(), y_obs, num_dim, yvar_obs_list=y_var, num_steps=10
    )

    assert model is not None
    assert isinstance(model, TurboGPNoisy)
    assert likelihood is not None
    assert isinstance(y_mean, float)
    assert isinstance(y_std, float)
    assert y_std > 0.0


def test_fit_gp_with_y_var_list_asserts_length():
    import numpy as np
    import pytest

    from enn.turbo.turbo_utils import fit_gp

    num_obs = 10
    num_dim = 2
    rng = np.random.default_rng(0)
    x_obs = rng.random((num_obs, num_dim)).tolist()
    y_obs = rng.random(num_obs).tolist()
    y_var_wrong_length = rng.uniform(0.01, 0.1, size=num_obs - 2).tolist()

    with pytest.raises(ValueError):
        fit_gp(x_obs, y_obs, num_dim, yvar_obs_list=y_var_wrong_length, num_steps=5)


def _make_turbo_gp_noisy(
    *,
    train_x,
    train_y,
    train_y_var,
    ard_dims: int,
    learn_additional_noise: bool = False,
):
    from gpytorch.constraints import Interval

    from enn.turbo.turbo_gp_noisy import TurboGPNoisy

    lengthscale_constraint = Interval(0.005, 2.0)
    outputscale_constraint = Interval(0.05, 20.0)
    return TurboGPNoisy(
        train_x=train_x,
        train_y=train_y,
        train_y_var=train_y_var,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
        learn_additional_noise=learn_additional_noise,
    )


def test_turbo_gp_noisy_accepts_train_y_var():
    import numpy as np
    import torch
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    num_obs = 10
    num_dim = 2
    rng = np.random.default_rng(42)
    train_x = torch.as_tensor(rng.random((num_obs, num_dim)), dtype=torch.float64)
    train_y = torch.as_tensor(
        train_x.sum(dim=1).numpy() + 0.1 * rng.standard_normal(num_obs),
        dtype=torch.float64,
    )
    train_y_var = torch.as_tensor(
        rng.uniform(0.01, 0.1, size=num_obs), dtype=torch.float64
    )
    model = _make_turbo_gp_noisy(
        train_x=train_x,
        train_y=train_y,
        train_y_var=train_y_var,
        ard_dims=num_dim,
    )

    assert model is not None
    assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
    assert model.covar_module is not None
    assert model.mean_module is not None


def test_turbo_gp_noisy_forward_and_posterior():
    import numpy as np
    import torch
    from gpytorch.distributions import MultivariateNormal

    num_obs = 15
    num_dim = 3
    rng = np.random.default_rng(123)
    train_x = torch.as_tensor(rng.random((num_obs, num_dim)), dtype=torch.float64)
    train_y = torch.as_tensor(
        train_x.sum(dim=1).numpy() + 0.05 * rng.standard_normal(num_obs),
        dtype=torch.float64,
    )
    train_y_var = torch.full((num_obs,), 0.01, dtype=torch.float64)

    model = _make_turbo_gp_noisy(
        train_x=train_x,
        train_y=train_y,
        train_y_var=train_y_var,
        ard_dims=num_dim,
    )

    model.eval()
    model.likelihood.eval()

    test_x = torch.as_tensor(rng.random((5, num_dim)), dtype=torch.float64)
    with torch.no_grad():
        forward_output = model.forward(test_x)
        posterior_output = model.posterior(test_x)

    assert isinstance(forward_output, MultivariateNormal)
    assert isinstance(posterior_output, MultivariateNormal)
    assert forward_output.mean.shape == (5,)
    assert posterior_output.mean.shape == (5,)


def test_turbo_gp_noisy_trains_successfully():
    import numpy as np
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood

    num_obs = 20
    num_dim = 2
    rng = np.random.default_rng(999)
    train_x = torch.as_tensor(rng.random((num_obs, num_dim)), dtype=torch.float64)
    train_y = torch.as_tensor(
        train_x.sum(dim=1).numpy() + 0.1 * rng.standard_normal(num_obs),
        dtype=torch.float64,
    )
    train_y_var = torch.as_tensor(
        rng.uniform(0.005, 0.05, size=num_obs), dtype=torch.float64
    )
    model = _make_turbo_gp_noisy(
        train_x=train_x,
        train_y=train_y,
        train_y_var=train_y_var,
        ard_dims=num_dim,
    )

    model.train()
    model.likelihood.train()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    initial_loss = None
    for i in range(20):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        if i == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss <= initial_loss


def test_turbo_gp_noisy_with_zero_variance():
    import numpy as np
    import torch
    import warnings

    num_obs = 10
    num_dim = 2
    rng = np.random.default_rng(42)
    train_x = torch.as_tensor(rng.random((num_obs, num_dim)), dtype=torch.float64)
    train_y = torch.as_tensor(train_x.sum(dim=1).numpy(), dtype=torch.float64)
    train_y_var = torch.zeros(num_obs, dtype=torch.float64)

    from gpytorch.utils.warnings import NumericalWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Very small noise values detected\..*",
            category=NumericalWarning,
        )
        model = _make_turbo_gp_noisy(
            train_x=train_x,
            train_y=train_y,
            train_y_var=train_y_var,
            ard_dims=num_dim,
            learn_additional_noise=True,
        )

    model.eval()
    model.likelihood.eval()

    test_x = torch.as_tensor(rng.random((3, num_dim)), dtype=torch.float64)
    with torch.no_grad():
        posterior = model.posterior(test_x)

    assert posterior.mean.shape == (3,)


def test_lhd_only_runs_after_initial_observations():
    import numpy as np

    from enn.turbo.turbo_config import LHDOnlyConfig
    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_optimizer import TurboOptimizer

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
    import numpy as np

    from enn.turbo.turbo_config import TurboENNConfig
    from enn.turbo.turbo_enn_impl import TurboENNImpl

    rng = np.random.default_rng(42)
    config = TurboENNConfig(k=5)
    impl = TurboENNImpl(config)

    result = impl.get_x_center([], [], rng)
    assert result is None

    x_obs = rng.random((20, 3)).tolist()
    y_obs = [float(i) for i in range(20)]
    result_before_fit = impl.get_x_center(x_obs, y_obs, rng)
    assert result_before_fit is not None
    x_array = np.asarray(x_obs, dtype=float)
    assert np.allclose(result_before_fit, x_array[19])

    impl.prepare_ask(x_obs, y_obs, [0.0] * 20, num_dim=3, gp_num_steps=10, rng=rng)
    result_after_fit = impl.get_x_center(x_obs, y_obs, rng)
    assert result_after_fit is not None
    assert result_after_fit.shape == (3,)
    top_5_indices = np.argsort(y_obs)[-5:]
    x_top_5 = x_array[top_5_indices]
    is_from_top_k = any(np.allclose(result_after_fit, x_top_5[i]) for i in range(5))
    assert is_from_top_k


def test_turbo_one_impl_get_x_center_requires_fit_for_n_ge_2():
    import numpy as np
    import pytest

    from enn.turbo.turbo_config import TurboOneConfig
    from enn.turbo.turbo_one_impl import TurboOneImpl

    rng = np.random.default_rng(0)
    impl = TurboOneImpl(TurboOneConfig())

    x1 = rng.random((1, 3)).tolist()
    y1 = [1.0]
    center1 = impl.get_x_center(x1, y1, rng)
    assert center1 is not None
    assert np.allclose(center1, np.asarray(x1, dtype=float)[0])

    x2 = rng.random((2, 3)).tolist()
    y2 = [0.0, 1.0]
    with pytest.raises(RuntimeError, match="prepare_ask"):
        impl.get_x_center(x2, y2, rng)

    impl.prepare_ask(x2, y2, [], num_dim=3, gp_num_steps=5, rng=rng)
    center2 = impl.get_x_center(x2, y2, rng)
    assert center2 is not None
    x2_array = np.asarray(x2, dtype=float)
    assert np.allclose(center2, x2_array[0]) or np.allclose(center2, x2_array[1])


def test_turbo_config_num_metrics_validation():
    import pytest

    from enn.turbo.turbo_config import TurboConfig

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


def test_turbo_optimizer_tell_without_yvar():
    import numpy as np

    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_optimizer import TurboOptimizer

    rng = np.random.default_rng(42)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = TurboOptimizer(bounds=bounds, mode=TurboMode.TURBO_ENN, rng=rng)

    x0 = opt.ask(num_arms=4)
    y0 = -np.sum(x0**2, axis=1)
    opt.tell(x0, y0)  # No y_var provided

    x1 = opt.ask(num_arms=4)
    y1 = -np.sum(x1**2, axis=1)
    opt.tell(x1, y1)  # No y_var provided

    x2 = opt.ask(num_arms=4)
    assert x2.shape == (4, 2)
    assert np.all(x2 >= 0.0) and np.all(x2 <= 1.0)


def test_turbo_optimizer_yvar_policy_enforced():
    import numpy as np

    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_optimizer import TurboOptimizer

    rng = np.random.default_rng(0)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)

    opt = TurboOptimizer(bounds=bounds, mode=TurboMode.TURBO_ONE, rng=rng)
    x0 = opt.ask(num_arms=2)
    y0 = -np.sum(x0**2, axis=1)
    yvar0 = 0.1 * np.ones_like(y0)
    opt.tell(x0, y0, yvar0)

    x1 = opt.ask(num_arms=2)
    y1 = -np.sum(x1**2, axis=1)
    with pytest.raises(ValueError, match="y_var must be provided"):
        opt.tell(x1, y1)

    rng = np.random.default_rng(0)
    opt2 = TurboOptimizer(bounds=bounds, mode=TurboMode.TURBO_ONE, rng=rng)
    x0 = opt2.ask(num_arms=2)
    y0 = -np.sum(x0**2, axis=1)
    opt2.tell(x0, y0)

    x1 = opt2.ask(num_arms=2)
    y1 = -np.sum(x1**2, axis=1)
    yvar1 = 0.1 * np.ones_like(y1)
    with pytest.raises(ValueError, match="y_var must be omitted"):
        opt2.tell(x1, y1, yvar1)


def test_turbo_one_trust_region_update_is_noise_robust_to_spikes():
    import numpy as np

    from enn.turbo.turbo_config import TurboOneConfig
    from enn.turbo.turbo_optimizer import TurboOptimizer

    rng = np.random.default_rng(0)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ONE,
        rng=rng,
        config=TurboOneConfig(num_init=1, num_candidates=16),
    )

    opt.ask(num_arms=1)
    x0 = np.zeros((1, 2), dtype=float)
    opt.tell(x0, np.array([0.0], dtype=float), y_var=np.array([1e6], dtype=float))

    opt.ask(num_arms=1)
    x1 = np.ones((1, 2), dtype=float)
    opt.tell(x1, np.array([100.0], dtype=float), y_var=np.array([1e6], dtype=float))

    assert opt._tr_state.best_value < 100.0
    assert opt._tr_state.best_value == max(opt._y_tr_list)


def test_turbo_enn_tr_values_use_posterior_mean_over_all_obs():
    import numpy as np

    from enn.turbo.turbo_config import TurboENNConfig
    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_optimizer import TurboOptimizer

    rng = np.random.default_rng(0)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ENN,
        rng=rng,
        config=TurboENNConfig(num_init=1, num_candidates=16, acq_type="pareto", k=3),
    )

    opt.ask(num_arms=1)
    x0 = np.zeros((1, 2), dtype=float)
    y0 = np.array([0.0], dtype=float)
    opt.tell(x0, y0, y_var=np.array([1e6], dtype=float))

    opt.ask(num_arms=1)
    x1 = np.ones((1, 2), dtype=float)
    y1 = np.array([100.0], dtype=float)
    opt.tell(x1, y1, y_var=np.array([1e6], dtype=float))

    # With huge observation noise, the ENN posterior mean at observed points should
    # be strongly smoothed and not match raw y.
    tr_vals = np.asarray(opt._y_tr_list, dtype=float)
    assert tr_vals.shape == (2,)
    assert not np.allclose(tr_vals, np.array([0.0, 100.0], dtype=float))
    assert np.all(tr_vals > 1.0) and np.all(tr_vals < 99.0)


def test_turbo_optimizer_no_trust_region_bounds_are_full_box():
    import numpy as np

    from enn.turbo.turbo_config import TurboZeroConfig
    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_optimizer import TurboOptimizer

    rng = np.random.default_rng(0)
    bounds = np.array([[-2.0, 2.0], [-1.0, 1.0], [0.0, 3.0]], dtype=float)
    opt = TurboOptimizer(
        bounds=bounds,
        mode=TurboMode.TURBO_ZERO,
        rng=rng,
        config=TurboZeroConfig(tr_type="none", num_init=1, num_candidates=16),
    )
    x0 = opt.ask(num_arms=1)
    opt.tell(x0, np.array([1.0]))
    x_center = np.array([0.25, 0.5, 0.75], dtype=float)
    lb, ub = opt._tr_state.compute_bounds_1d(x_center)
    assert np.allclose(lb, 0.0)
    assert np.allclose(ub, 1.0)


def test_morbo_chebyshev_trust_region_weights_and_scaling():
    import numpy as np
    from scipy.stats import qmc

    from enn.turbo.morbo_trust_region import MorboTrustRegion

    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    tr1 = MorboTrustRegion(num_dim=3, num_arms=1, num_metrics=2, rng=rng1)
    tr2 = MorboTrustRegion(num_dim=3, num_arms=1, num_metrics=2, rng=rng2)
    assert np.allclose(tr1.weights, tr2.weights)
    assert tr1.weights.shape == (2,)
    assert np.all(tr1.weights > 0.0)
    assert np.isclose(tr1.weights.sum(), 1.0)

    x_obs = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]], dtype=float)
    y_obs = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    tr1.update_xy(x_obs, y_obs)
    scores = tr1.scalarize(y_obs, clip=True)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
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
    assert x_cand.shape == (64, 3)
    assert np.all(x_cand >= 0.0) and np.all(x_cand <= 1.0)


def test_turbo_optimizer_morbo_multi_objective():
    import numpy as np

    from enn.turbo.turbo_config import TurboENNConfig
    from enn.turbo.turbo_mode import TurboMode
    from enn.turbo.turbo_optimizer import TurboOptimizer

    num_dim = 3
    num_metrics = 2
    bounds = np.array([[0.0, 1.0]] * num_dim, dtype=float)
    rng = np.random.default_rng(42)

    config = TurboENNConfig(tr_type="morbo", num_metrics=num_metrics, num_init=4)
    optimizer = TurboOptimizer(
        bounds=bounds, mode=TurboMode.TURBO_ENN, rng=rng, config=config
    )

    # Initial ask/tell loop
    for _ in range(3):
        x = optimizer.ask(num_arms=2)
        assert x.shape == (2, num_dim)
        # 2D y with shape (n, num_metrics)
        y = rng.uniform(0.0, 1.0, size=(2, num_metrics))
        y_est = optimizer.tell(x, y)
        # y_est should be 2D with shape (n, num_metrics)
        assert y_est.shape == (2, num_metrics)

    # After init: should use normal ask with trust region
    x = optimizer.ask(num_arms=2)
    assert x.shape == (2, num_dim)
    y = rng.uniform(0.0, 1.0, size=(2, num_metrics))
    y_est = optimizer.tell(x, y)
    # After ENN is fit, y_est is still 2D with shape (n, num_metrics)
    assert y_est.shape == (2, num_metrics)

    # Verify trust region length is set
    tr_len = optimizer.tr_length
    assert tr_len is not None
    assert 0.0 < tr_len <= 1.0


def test_fit_gp_multi_output_can_trigger_non_scalar_backward_error():
    import numpy as np
    import pytest
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from enn.turbo.turbo_utils import fit_gp

    rng = np.random.default_rng(0)
    num_dim = 3
    num_metrics = 2
    n = 8

    x = rng.uniform(0.0, 1.0, size=(n, num_dim))
    y = rng.normal(size=(n, num_metrics))
    model, likelihood, _gp_y_mean, _gp_y_std = fit_gp(
        x.tolist(), y.tolist(), num_dim, num_steps=0
    )
    assert model is not None
    assert likelihood is not None

    model.train()
    likelihood.train()
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    output = model(train_x)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    loss = -mll(output, train_y)
    assert tuple(loss.shape) == (num_metrics,)
    with pytest.raises(
        RuntimeError, match="grad can be implicitly created only for scalar outputs"
    ):
        loss.backward()


def test_fit_gp_multi_output_trains_without_scalar_backward_error():
    import numpy as np

    from enn.turbo.turbo_utils import fit_gp

    rng = np.random.default_rng(0)
    num_dim = 3
    num_metrics = 2
    n = 8

    x = rng.uniform(0.0, 1.0, size=(n, num_dim))
    y = rng.normal(size=(n, num_metrics))
    model, likelihood, gp_y_mean, gp_y_std = fit_gp(
        x.tolist(), y.tolist(), num_dim, num_steps=2
    )
    assert model is not None
    assert likelihood is not None
    assert gp_y_mean.shape == (num_metrics,)
    assert gp_y_std.shape == (num_metrics,)
