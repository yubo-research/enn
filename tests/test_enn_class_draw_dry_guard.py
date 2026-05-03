"""Regression guards for EpistemicNearestNeighbors function-draw paths before/after DRY refactors."""

from __future__ import annotations

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


def _tiny_model():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    train_y = np.array([[0.1], [0.2], [0.3]], dtype=float)
    return EpistemicNearestNeighbors(train_x, train_y)


def test_posterior_function_draw_flags_none_matches_explicit_default():
    model = _tiny_model()
    x = np.array([[0.25, 0.25]], dtype=float)
    params = ENNParams(2, 1.0, 0.1)
    seeds = np.array([7, 8, 9], dtype=np.int64)
    d1, i1 = model.posterior_function_draw(x, params, function_seeds=seeds, flags=None)
    d2, i2 = model.posterior_function_draw(
        x, params, function_seeds=seeds, flags=PosteriorFlags()
    )
    assert np.allclose(d1, d2)
    assert np.array_equal(np.asarray(i1, dtype=int), np.asarray(i2, dtype=int))


def test_posterior_function_draw_repeatable():
    model = _tiny_model()
    x = np.array([[0.25, 0.25], [0.5, 0.1]], dtype=float)
    params = ENNParams(2, 1.0, 0.1)
    flags = PosteriorFlags(exclude_nearest=True, observation_noise=False)
    seeds = [1001, 1002]
    a1, ix1 = model.posterior_function_draw(
        x, params, function_seeds=seeds, flags=flags
    )
    a2, ix2 = model.posterior_function_draw(
        x, params, function_seeds=seeds, flags=flags
    )
    assert np.allclose(a1, a2)
    assert np.array_equal(np.asarray(ix1, dtype=int), np.asarray(ix2, dtype=int))


def test_function_draw_idx_shape_dtype():
    model = _tiny_model()
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    params = ENNParams(2, 1.0, 0.1)
    seeds = [1]
    draws, idx = model.posterior_function_draw(
        x, params, function_seeds=seeds, flags=PosteriorFlags()
    )
    assert draws.shape == (1, x.shape[0], model.num_outputs)
    idx_arr = np.asarray(idx, dtype=int)
    assert idx_arr.shape == (x.shape[0], params.k_num_neighbors)


def test_conditional_function_draw_empty_whatif_matches_posterior_golden():
    """Locks parity between delegation branch and posterior_function_draw."""
    model = _tiny_model()
    x = np.array([[0.11, 0.22]], dtype=float)
    params = ENNParams(2, 1.0, 0.05)
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=True)
    seeds = np.array([3, 4], dtype=np.int64)
    d_post, i_post = model.posterior_function_draw(
        x, params, function_seeds=seeds, flags=flags
    )
    d_cond, i_cond = model.conditional_posterior_function_draw(
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 1), dtype=float),
        x,
        params=params,
        function_seeds=seeds,
        flags=flags,
    )
    assert np.allclose(d_post, d_cond, rtol=0.0, atol=0.0)
    assert np.array_equal(np.asarray(i_post, dtype=int), np.asarray(i_cond, dtype=int))


def test_function_draw_golden_values_fixed_seed_arrays():
    """Byte-level guard on one deterministic draw (update only if ENN math intentionally changes)."""
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    model = EpistemicNearestNeighbors(train_x, train_y)
    x = np.array([[0.25, 0.25]], dtype=float)
    params = ENNParams(2, 1.0, 0.1)
    seeds = np.array([42, 43], dtype=np.int64)
    draws, _idx = model.posterior_function_draw(
        x, params, function_seeds=seeds, flags=PosteriorFlags()
    )
    # Expected from enn EpistemicNearestNeighbors + Rust draw path (fixed data/seeds).
    expected = np.array(
        [[[-0.71338448]], [[-0.40793862]]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(draws, expected, rtol=1e-7, atol=1e-7)
