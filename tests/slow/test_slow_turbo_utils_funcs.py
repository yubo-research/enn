from __future__ import annotations

import conftest
import numpy as np
import pytest
import turbo_utils_selection_helpers as selection_helpers

from enn.turbo.python_fallback.turbo_utils import gp_thompson_sample


@pytest.mark.slow
def test_gp_thompson_sample_returns_valid_indices():
    from enn.turbo.python_fallback.turbo_gp_fit import fit_gp

    num_obs, num_dim = 10, 2
    rng = np.random.default_rng(42)
    x_obs = rng.random((num_obs, num_dim))
    y_obs = x_obs.sum(axis=1) + 0.1 * rng.standard_normal(num_obs)
    result = fit_gp(x_obs.tolist(), y_obs.tolist(), num_dim, num_steps=2)
    x_cand = rng.random((20, num_dim))
    num_arms = 3
    y_mean = float(np.mean(y_obs))
    y_std = float(np.std(y_obs))
    indices = gp_thompson_sample(
        result.model, x_cand, num_arms, rng, gp_y_mean=y_mean, gp_y_std=y_std
    )
    assert len(indices) == num_arms
    assert all(0 <= i < len(x_cand) for i in indices)


@pytest.mark.slow
def test_select_gp_thompson_uses_gp_and_returns_correct_shape():
    num_candidates, num_dim, num_arms = 30, 2, 5
    x_cand = np.random.default_rng(0).random((num_candidates, num_dim))
    x_obs = np.random.default_rng(1).random((15, num_dim))
    y_obs = x_obs.sum(axis=1).tolist()
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    from_unit_fn = conftest.make_from_unit_fn(bounds)
    select_sobol_fn = conftest.make_select_sobol_fn(bounds, rng)
    selected, (new_mean, new_std), _ = selection_helpers.select_gp_thompson(
        x_cand,
        num_arms,
        x_obs.tolist(),
        y_obs,
        num_dim,
        gp_num_steps=2,
        rng=rng,
        gp_y_stats=(0.0, 1.0),
        select_sobol_fn=select_sobol_fn,
        from_unit_fn=from_unit_fn,
    )
    assert selected.shape == (num_arms, num_dim)
    assert isinstance(new_mean, float) and isinstance(new_std, float)
    assert new_std > 0.0
    assert np.all(selected >= bounds[:, 0]) and np.all(selected <= bounds[:, 1])
