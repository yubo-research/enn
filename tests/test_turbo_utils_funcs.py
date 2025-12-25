from __future__ import annotations

import conftest
import numpy as np
import pytest
from scipy.stats import qmc

from enn.turbo.turbo_utils import (
    argmax_random_tie,
    from_unit,
    latin_hypercube,
    raasp,
    sobol_perturb_np,
    to_unit,
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
    x_center = np.full(num_dim, 0.5)
    lb, ub = np.zeros(num_dim), np.ones(num_dim)
    mask = np.zeros((num_candidates, num_dim), dtype=bool)
    mask[:, 0] = True
    mask[0, 1] = True
    sobol_engine = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    candidates = sobol_perturb_np(
        x_center, lb, ub, num_candidates, mask, sobol_engine=sobol_engine
    )
    for i in range(num_candidates):
        for j in range(num_dim):
            if mask[i, j]:
                assert candidates[i, j] != x_center[j]
            else:
                assert candidates[i, j] == x_center[j]


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


def test_raasp_shape_and_bounds():
    num_candidates, num_dim = 10, 3
    x_center = np.full(num_dim, 0.5)
    lb, ub = np.zeros(num_dim), np.ones(num_dim)
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
    assert np.all(candidates >= lb) and np.all(candidates <= ub)


def test_raasp_at_least_one_dimension_perturbed():
    num_candidates, num_dim = 20, 5
    x_center = np.full(num_dim, 0.5)
    lb, ub = np.zeros(num_dim), np.ones(num_dim)
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
        assert np.any(np.abs(candidates[i] - x_center) > 1e-10)


def test_raasp_deterministic():
    num_candidates, num_dim = 8, 2
    x_center = np.full(num_dim, 0.5)
    lb, ub = np.zeros(num_dim), np.ones(num_dim)
    rng1, rng2 = np.random.default_rng(42), np.random.default_rng(42)
    sobol1 = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    sobol2 = qmc.Sobol(d=num_dim, scramble=True, seed=0)
    c1 = raasp(
        x_center, lb, ub, num_candidates, num_pert=20, rng=rng1, sobol_engine=sobol1
    )
    c2 = raasp(
        x_center, lb, ub, num_candidates, num_pert=20, rng=rng2, sobol_engine=sobol2
    )
    assert np.allclose(c1, c2)


def test_raasp_probability_scaling():
    num_candidates = 100
    num_dim_low, num_dim_high = 5, 100
    x_low, x_high = np.full(num_dim_low, 0.5), np.full(num_dim_high, 0.5)
    lb_low, ub_low = np.zeros(num_dim_low), np.ones(num_dim_low)
    lb_high, ub_high = np.zeros(num_dim_high), np.ones(num_dim_high)
    rng = np.random.default_rng(0)
    sobol_low = qmc.Sobol(d=num_dim_low, scramble=True, seed=0)
    sobol_high = qmc.Sobol(d=num_dim_high, scramble=True, seed=0)
    c_low = raasp(
        x_low,
        lb_low,
        ub_low,
        num_candidates,
        num_pert=20,
        rng=rng,
        sobol_engine=sobol_low,
    )
    rng = np.random.default_rng(0)
    c_high = raasp(
        x_high,
        lb_high,
        ub_high,
        num_candidates,
        num_pert=20,
        rng=rng,
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
