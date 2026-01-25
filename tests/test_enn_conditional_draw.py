from __future__ import annotations
import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


def test_conditional_posterior_function_draw_matches_unconditional_when_no_whatifs():
    rng = np.random.default_rng(0)
    train_x = rng.standard_normal((20, 3))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(train_x, train_y, 0.1 * np.ones_like(train_y))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=True)
    seeds = [1, 2, 3]
    draws = model.posterior_function_draw(
        x_test, params, function_seeds=seeds, flags=flags
    )
    draws_cond = model.conditional_posterior_function_draw(
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 1), dtype=float),
        x_test,
        params=params,
        function_seeds=seeds,
        flags=flags,
    )
    assert np.allclose(draws, draws_cond)


def test_conditional_posterior_function_draw_is_deterministic_and_does_not_mutate_index():
    d = 3
    train_x = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=float)
    train_y = np.array([[0.0], [0.0]], dtype=float)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar=None)
    x_test = np.zeros((1, d), dtype=float)
    x_whatif = np.zeros((1, d), dtype=float)
    y_whatif = np.array([[5.0]], dtype=float)
    neighbors_before = model.neighbors(x_test, k=2, exclude_nearest=False)
    params = ENNParams(
        k_num_neighbors=1, epistemic_variance_scale=1.0, aleatoric_variance_scale=1.0
    )
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=True)
    seeds = [123, 124]
    draws1 = model.conditional_posterior_function_draw(
        x_whatif,
        y_whatif,
        x_test,
        params=params,
        function_seeds=seeds,
        flags=flags,
    )
    draws2 = model.conditional_posterior_function_draw(
        x_whatif,
        y_whatif,
        x_test,
        params=params,
        function_seeds=seeds,
        flags=flags,
    )
    neighbors_after = model.neighbors(x_test, k=2, exclude_nearest=False)
    assert len(neighbors_before) == len(neighbors_after)
    for (xb, yb), (xa, ya) in zip(neighbors_before, neighbors_after):
        assert np.allclose(xb, xa)
        assert np.allclose(yb, ya)
    assert draws1.shape == (len(seeds), 1, 1)
    assert np.allclose(draws1, draws2)


def test_conditional_posterior_function_draw_changes_with_whatifs_and_exclude_nearest():
    d = 3
    train_x = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=float)
    train_y = np.array([[1.0], [2.0]], dtype=float)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar=None)
    x_test = np.zeros((1, d), dtype=float)
    x_whatif = np.zeros((1, d), dtype=float)
    y_whatif = np.array([[100.0]], dtype=float)
    params = ENNParams(
        k_num_neighbors=1, epistemic_variance_scale=1.0, aleatoric_variance_scale=1.0
    )
    seeds = [7]
    draw_base = model.posterior_function_draw(x_test, params, function_seeds=seeds)[
        0, 0, 0
    ]
    draw_incl = model.conditional_posterior_function_draw(
        x_whatif,
        y_whatif,
        x_test,
        params=params,
        function_seeds=seeds,
        flags=PosteriorFlags(exclude_nearest=False, observation_noise=True),
    )[0, 0, 0]
    draw_excl = model.conditional_posterior_function_draw(
        x_whatif,
        y_whatif,
        x_test,
        params=params,
        function_seeds=seeds,
        flags=PosteriorFlags(exclude_nearest=True, observation_noise=True),
    )[0, 0, 0]
    assert not np.allclose(draw_base, draw_incl)
    assert not np.allclose(draw_incl, draw_excl)


def test_conditional_posterior_function_draw_matches_augmented_model_exactly():
    rng = np.random.default_rng(0)
    n_train, n_whatif, d = 8, 3, 3
    train_x = rng.standard_normal((n_train, d))
    train_y = train_x.sum(axis=1, keepdims=True).astype(float)
    x_whatif = rng.standard_normal((n_whatif, d))
    y_whatif = x_whatif.sum(axis=1, keepdims=True).astype(float)
    x_test = rng.standard_normal((5, d))
    params = ENNParams(
        k_num_neighbors=4, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    flags = PosteriorFlags(exclude_nearest=True, observation_noise=True)
    function_seeds = [11, 12, 13]
    enn_a = EpistemicNearestNeighbors(train_x, train_y, train_yvar=None)
    draws_a = enn_a.conditional_posterior_function_draw(
        x_whatif,
        y_whatif,
        x_test,
        params=params,
        function_seeds=function_seeds,
        flags=flags,
    )
    enn_b = EpistemicNearestNeighbors(
        np.concatenate([train_x, x_whatif], axis=0),
        np.concatenate([train_y, y_whatif], axis=0),
        train_yvar=None,
    )
    draws_b = enn_b.posterior_function_draw(
        x_test, params, function_seeds=function_seeds, flags=flags
    )
    assert np.allclose(draws_a, draws_b)
