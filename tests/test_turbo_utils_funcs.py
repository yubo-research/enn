from __future__ import annotations
import conftest
import numpy as np
import pytest
from scipy.stats import qmc
from enn.turbo.turbo_utils import (
    argmax_random_tie,
    from_unit,
    generate_raasp_candidates,
    generate_raasp_candidates_uniform,
    gp_thompson_sample,
    latin_hypercube,
    raasp_perturb,
    record_duration,
    sobol_perturb_np,
    to_unit,
    uniform_perturb_np,
)


def test_latin_hypercube_stratification_and_bounds():
    rng = np.random.default_rng(0)
    n, d = 8, 3
    x = latin_hypercube(n, d, rng=rng)
    assert x.shape == (n, d)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    for j in range(d):
        xs = np.sort(x[:, j])
        for k in range(n):
            assert np.any((xs >= k / n) & (xs <= (k + 1) / n + 1e-8))


def test_argmax_random_tie_uses_rng_and_is_deterministic():
    values = np.array([1.0, 2.0, 2.0, 0.0], dtype=float)
    rng = np.random.default_rng(0)
    idx1 = argmax_random_tie(values, rng=rng)
    assert idx1 in (1, 2)
    rng = np.random.default_rng(0)
    idx2 = argmax_random_tie(values, rng=rng)
    assert idx1 == idx2


def test_record_duration_sets_dt():
    dt_holder: list[float] = []

    def set_dt(dt: float) -> None:
        dt_holder.append(float(dt))

    with record_duration(set_dt):
        pass
    assert len(dt_holder) == 1
    assert dt_holder[0] >= 0.0


def test_sobol_perturb_np_shape_and_bounds():
    num_candidates, num_dim = 10, 3
    x_center = np.full(num_dim, 0.5)
    lb, ub = np.zeros(num_dim), np.ones(num_dim)
    mask = np.ones((num_candidates, num_dim), dtype=bool)
    sobol_engine = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    candidates = sobol_perturb_np(
        x_center, lb, ub, num_candidates, mask, sobol_engine=sobol_engine
    )
    assert candidates.shape == (num_candidates, num_dim)
    assert np.all(candidates >= lb) and np.all(candidates <= ub)


def test_sobol_perturb_np_mask_application():
    num_candidates, num_dim = 5, 3
    x_center, lb, ub = np.full(num_dim, 0.5), np.zeros(num_dim), np.ones(num_dim)
    mask = np.zeros((num_candidates, num_dim), dtype=bool)
    mask[:, 0], mask[0, 1] = True, True
    candidates = sobol_perturb_np(
        x_center,
        lb,
        ub,
        num_candidates,
        mask,
        sobol_engine=qmc.Sobol(d=num_dim, scramble=True, seed=0),
    )
    for i in range(num_candidates):
        for j in range(num_dim):
            assert (
                (candidates[i, j] != x_center[j])
                if mask[i, j]
                else (candidates[i, j] == x_center[j])
            )


def test_sobol_perturb_np_deterministic():
    num_candidates, num_dim = 8, 2
    x_center = np.full(num_dim, 0.5)
    lb, ub = np.zeros(num_dim), np.ones(num_dim)
    mask = np.ones((num_candidates, num_dim), dtype=bool)
    sobol1 = qmc.Sobol(d=num_dim, scramble=True, seed=42)
    sobol2 = qmc.Sobol(d=num_dim, scramble=True, seed=42)
    c1 = sobol_perturb_np(x_center, lb, ub, num_candidates, mask, sobol_engine=sobol1)
    c2 = sobol_perturb_np(x_center, lb, ub, num_candidates, mask, sobol_engine=sobol2)
    assert np.allclose(c1, c2)


def _raasp_perturb_test(rng, candidate_rv, num_dim=3, sobol_engine=None):
    num_candidates = 10
    x_center, lb, ub = np.full(num_dim, 0.5), np.zeros(num_dim), np.ones(num_dim)
    candidates = raasp_perturb(
        x_center,
        lb,
        ub,
        num_candidates,
        num_pert=20,
        rng=rng,
        candidate_rv=candidate_rv,
        sobol_engine=sobol_engine,
    )
    _check_candidate_shape_and_bounds(candidates, num_candidates, num_dim, lb, ub)
    return candidates, x_center


def test_raasp_shape_and_bounds():
    from enn.turbo.config import CandidateRV

    _raasp_perturb_test(
        np.random.default_rng(0),
        CandidateRV.SOBOL,
        num_dim=3,
        sobol_engine=qmc.Sobol(d=3, scramble=True, seed=0),
    )


def test_raasp_at_least_one_dimension_perturbed():
    from enn.turbo.config import CandidateRV

    candidates, x_center = _raasp_perturb_test(
        np.random.default_rng(0),
        CandidateRV.SOBOL,
        num_dim=5,
        sobol_engine=qmc.Sobol(d=5, scramble=True, seed=0),
    )
    for i in range(len(candidates)):
        assert np.any(np.abs(candidates[i] - x_center) > 1e-10)


def test_raasp_deterministic():
    from enn.turbo.config import CandidateRV

    num_candidates, num_dim = 8, 2
    x_center = np.full(num_dim, 0.5)
    lb, ub = np.zeros(num_dim), np.ones(num_dim)
    rng1, rng2 = np.random.default_rng(42), np.random.default_rng(42)
    sobol1 = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    sobol2 = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    c1 = raasp_perturb(
        x_center,
        lb,
        ub,
        num_candidates,
        num_pert=20,
        rng=rng1,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol1,
    )
    c2 = raasp_perturb(
        x_center,
        lb,
        ub,
        num_candidates,
        num_pert=20,
        rng=rng2,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol2,
    )
    assert np.allclose(c1, c2)


def test_raasp_probability_scaling():
    from enn.turbo.config import CandidateRV

    num_candidates = 100
    num_dim_low, num_dim_high = 5, 100
    x_low, x_high = np.full(num_dim_low, 0.5), np.full(num_dim_high, 0.5)
    lb_low, ub_low = np.zeros(num_dim_low), np.ones(num_dim_low)
    lb_high, ub_high = np.zeros(num_dim_high), np.ones(num_dim_high)
    rng = np.random.default_rng(0)
    sobol_low = qmc.Sobol(d=num_dim_low, scramble=True, seed=0)
    sobol_high = qmc.Sobol(d=num_dim_high, scramble=True, seed=0)
    c_low = raasp_perturb(
        x_low,
        lb_low,
        ub_low,
        num_candidates,
        num_pert=20,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol_low,
    )
    rng = np.random.default_rng(0)
    c_high = raasp_perturb(
        x_high,
        lb_high,
        ub_high,
        num_candidates,
        num_pert=20,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol_high,
    )
    diff_low = np.sum(np.abs(c_low - x_low) > 1e-10, axis=1)
    diff_high = np.sum(np.abs(c_high - x_high) > 1e-10, axis=1)
    assert np.mean(diff_low) / num_dim_low > np.mean(diff_high) / num_dim_high


def test_to_unit_and_from_unit_roundtrip():
    bounds = np.array([[0.0, 2.0], [-1.0, 1.0], [5.0, 10.0]], dtype=float)
    x_original = np.array([[1.0, 0.0, 7.5], [0.5, -0.5, 8.0]], dtype=float)
    x_unit = to_unit(x_original, bounds)
    assert x_unit.shape == x_original.shape
    assert np.all(x_unit >= 0.0) and np.all(x_unit <= 1.0)
    assert np.allclose(x_original, from_unit(x_unit, bounds))


def test_to_unit_bounds_validation():
    bounds_invalid = np.array([[1.0, 0.0]], dtype=float)
    x = np.array([[0.5]], dtype=float)
    with pytest.raises(ValueError):
        to_unit(x, bounds_invalid)


def test_select_uniform_shape_and_uniformity():
    from enn.turbo.proposal import select_uniform

    num_candidates, num_dim, num_arms = 128, 4, 8
    x_cand = np.random.default_rng(0).random((num_candidates, num_dim))
    bounds = np.array([[0.0, 1.0]] * num_dim, dtype=float)
    rng = np.random.default_rng(42)
    from_unit_fn = conftest.make_from_unit_fn(bounds)
    selected = select_uniform(x_cand, num_arms, num_dim, rng, from_unit_fn)
    assert selected.shape == (num_arms, num_dim)
    assert len(np.unique([tuple(row) for row in selected], axis=0)) == num_arms


def test_select_uniform_validation():
    from enn.turbo.proposal import select_uniform

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    from_unit_fn = conftest.make_from_unit_fn(bounds)
    with pytest.raises(ValueError):
        select_uniform(
            np.random.default_rng(0).random((10, 3)), 5, 2, rng, from_unit_fn
        )
    with pytest.raises(ValueError):
        select_uniform(np.random.default_rng(0).random((3, 2)), 5, 2, rng, from_unit_fn)


def test_select_gp_thompson_uses_gp_and_returns_correct_shape():
    from enn.turbo.proposal import select_gp_thompson

    num_candidates, num_dim, num_arms = 30, 2, 5
    x_cand = np.random.default_rng(0).random((num_candidates, num_dim))
    x_obs = np.random.default_rng(1).random((15, num_dim))
    y_obs = x_obs.sum(axis=1).tolist()
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    from_unit_fn = conftest.make_from_unit_fn(bounds)
    select_sobol_fn = conftest.make_select_sobol_fn(bounds, rng)
    selected, (new_mean, new_std), _ = select_gp_thompson(
        x_cand,
        num_arms,
        x_obs.tolist(),
        y_obs,
        num_dim,
        gp_num_steps=20,
        rng=rng,
        gp_y_stats=(0.0, 1.0),
        select_sobol_fn=select_sobol_fn,
        from_unit_fn=from_unit_fn,
    )
    assert selected.shape == (num_arms, num_dim)
    assert isinstance(new_mean, float) and isinstance(new_std, float)
    assert new_std > 0.0
    assert np.all(selected >= bounds[:, 0]) and np.all(selected <= bounds[:, 1])


def test_select_gp_thompson_fallback_on_empty_observations():
    from enn.turbo.proposal import select_gp_thompson

    num_candidates, num_dim, num_arms = 20, 2, 3
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

    selected, (mean, std), _ = select_gp_thompson(
        x_cand,
        num_arms,
        [],
        [],
        num_dim,
        gp_num_steps=20,
        rng=rng,
        gp_y_stats=(0.0, 1.0),
        select_sobol_fn=select_sobol_fn,
        from_unit_fn=from_unit_fn,
    )
    assert fallback_called
    assert selected.shape == (num_arms, num_dim)
    assert mean == 0.0 and std == 1.0


def test_uniform_perturb_np_shape_and_bounds():
    num_candidates, num_dim = 10, 3
    x_center = np.full(num_dim, 0.5)
    lb, ub = np.zeros(num_dim), np.ones(num_dim)
    mask = np.ones((num_candidates, num_dim), dtype=bool)
    rng = np.random.default_rng(0)
    candidates = uniform_perturb_np(x_center, lb, ub, num_candidates, mask, rng=rng)
    assert candidates.shape == (num_candidates, num_dim)
    assert np.all(candidates >= lb) and np.all(candidates <= ub)


def test_uniform_perturb_np_mask_application():
    num_candidates, num_dim = 5, 3
    x_center, lb, ub = np.full(num_dim, 0.5), np.zeros(num_dim), np.ones(num_dim)
    mask = np.zeros((num_candidates, num_dim), dtype=bool)
    mask[:, 0] = True
    candidates = uniform_perturb_np(
        x_center, lb, ub, num_candidates, mask, rng=np.random.default_rng(0)
    )
    for i in range(num_candidates):
        for j in range(num_dim):
            if not mask[i, j]:
                assert candidates[i, j] == x_center[j]


def _check_candidate_shape_and_bounds(candidates, num_candidates, num_dim, lb, ub):
    assert candidates.shape == (num_candidates, num_dim)
    assert np.all(candidates >= lb) and np.all(candidates <= ub)


def test_raasp_uniform_shape_and_bounds():
    from enn.turbo.config import CandidateRV

    _raasp_perturb_test(np.random.default_rng(0), CandidateRV.UNIFORM)


def test_raasp_uniform_at_least_one_dimension_perturbed():
    from enn.turbo.config import CandidateRV

    candidates, x_center = _raasp_perturb_test(
        np.random.default_rng(0), CandidateRV.UNIFORM, num_dim=5
    )
    for i in range(len(candidates)):
        assert np.any(np.abs(candidates[i] - x_center) > 1e-10)


def test_generate_raasp_candidates_shape_and_bounds():
    from enn.turbo.config import CandidateRV

    num_candidates, num_dim = 10, 3
    x_center, lb, ub = np.full(num_dim, 0.5), np.zeros(num_dim), np.ones(num_dim)
    rng = np.random.default_rng(42)
    sobol = qmc.Sobol(d=num_dim, scramble=True, seed=42)
    candidates = generate_raasp_candidates(
        x_center,
        lb,
        ub,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol,
    )
    _check_candidate_shape_and_bounds(candidates, num_candidates, num_dim, lb, ub)


def test_generate_raasp_candidates_uniform_shape_and_bounds():
    num_candidates, num_dim = 10, 3
    x_center, lb, ub = np.full(num_dim, 0.5), np.zeros(num_dim), np.ones(num_dim)
    rng = np.random.default_rng(42)
    candidates = generate_raasp_candidates_uniform(
        x_center, lb, ub, num_candidates, rng=rng
    )
    _check_candidate_shape_and_bounds(candidates, num_candidates, num_dim, lb, ub)


def test_generate_raasp_candidates_uniform_respects_num_pert():
    num_candidates = 200
    num_dim_low, num_dim_high = 5, 50
    x_low = np.full(num_dim_low, 0.5)
    x_high = np.full(num_dim_high, 0.5)
    lb_low, ub_low = np.zeros(num_dim_low), np.ones(num_dim_low)
    lb_high, ub_high = np.zeros(num_dim_high), np.ones(num_dim_high)
    rng_low = np.random.default_rng(0)
    rng_high = np.random.default_rng(0)
    c_low_pert = generate_raasp_candidates_uniform(
        x_low, lb_low, ub_low, num_candidates, rng=rng_low, num_pert=1
    )
    c_high_pert = generate_raasp_candidates_uniform(
        x_low, lb_low, ub_low, num_candidates, rng=rng_high, num_pert=20
    )
    diff_low = np.mean(np.sum(np.abs(c_low_pert - x_low) > 1e-10, axis=1) / num_dim_low)
    diff_high = np.mean(
        np.sum(np.abs(c_high_pert - x_low) > 1e-10, axis=1) / num_dim_low
    )
    assert diff_high > diff_low + 0.05
    rng_low = np.random.default_rng(1)
    rng_high = np.random.default_rng(1)
    c_low_pert = generate_raasp_candidates_uniform(
        x_high, lb_high, ub_high, num_candidates, rng=rng_low, num_pert=1
    )
    c_high_pert = generate_raasp_candidates_uniform(
        x_high, lb_high, ub_high, num_candidates, rng=rng_high, num_pert=20
    )
    diff_low = np.mean(
        np.sum(np.abs(c_low_pert - x_high) > 1e-10, axis=1) / num_dim_high
    )
    diff_high = np.mean(
        np.sum(np.abs(c_high_pert - x_high) > 1e-10, axis=1) / num_dim_high
    )
    assert diff_high > diff_low + 0.05


def test_gp_thompson_sample_returns_valid_indices():
    from enn.turbo.turbo_gp_fit import fit_gp

    num_obs, num_dim = 10, 2
    rng = np.random.default_rng(42)
    x_obs = rng.random((num_obs, num_dim))
    y_obs = x_obs.sum(axis=1) + 0.1 * rng.standard_normal(num_obs)
    result = fit_gp(x_obs.tolist(), y_obs.tolist(), num_dim, num_steps=10)
    x_cand = rng.random((20, num_dim))
    num_arms = 3
    y_mean = float(np.mean(y_obs))
    y_std = float(np.std(y_obs))
    indices = gp_thompson_sample(
        result.model, x_cand, num_arms, rng, gp_y_mean=y_mean, gp_y_std=y_std
    )
    assert len(indices) == num_arms
    assert all(0 <= i < len(x_cand) for i in indices)


def test_next_power_of_2():
    def next_power_of_2(n):
        return 1 if n <= 0 else 1 << (n - 1).bit_length()

    assert next_power_of_2(0) == 1
    assert next_power_of_2(1) == 1
    assert next_power_of_2(2) == 2
    assert next_power_of_2(3) == 4
    assert next_power_of_2(4) == 4
    assert next_power_of_2(5) == 8
    assert next_power_of_2(7) == 8
    assert next_power_of_2(8) == 8
    assert next_power_of_2(9) == 16
    assert next_power_of_2(15) == 16
    assert next_power_of_2(16) == 16
    assert next_power_of_2(17) == 32
