from __future__ import annotations
import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


def test_conditional_posterior_matches_posterior_when_no_whatifs():
    import conftest

    model, _train_x, _train_y, _train_yvar, rng = conftest.make_enn_model()
    x_test = rng.standard_normal((6, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    flags = PosteriorFlags(exclude_nearest=True, observation_noise=True)
    x_whatif = np.zeros((0, 3), dtype=float)
    y_whatif = np.zeros((0, 1), dtype=float)
    post = model.posterior(x_test, params=params, flags=flags)
    post_cond = model.conditional_posterior(
        x_whatif, y_whatif, x_test, params=params, flags=flags
    )
    assert np.allclose(post.mu, post_cond.mu)
    assert np.allclose(post.se, post_cond.se)


def test_conditional_posterior_includes_whatif_points_and_does_not_mutate_index():
    d = 3
    train_x = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=float)
    train_y = np.array([[0.0], [0.0]], dtype=float)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar=None)
    x_test = np.zeros((1, d), dtype=float)
    x_whatif = np.zeros((1, d), dtype=float)
    y_whatif = np.array([[5.0]], dtype=float)
    neighbors_before = model.neighbors(x_test, k=2, exclude_nearest=False)
    post_cond = model.conditional_posterior(
        x_whatif,
        y_whatif,
        x_test,
        params=ENNParams(
            k_num_neighbors=1,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.0,
        ),
        flags=PosteriorFlags(exclude_nearest=False, observation_noise=False),
    )
    neighbors_after = model.neighbors(x_test, k=2, exclude_nearest=False)
    assert np.all(neighbors_before == neighbors_after)
    assert np.allclose(post_cond.mu, 5.0)


def test_conditional_posterior_exclude_nearest_drops_whatif_point():
    d = 3
    train_x = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=float)
    train_y = np.array([[1.0], [2.0]], dtype=float)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar=train_yvar)
    x_test = np.zeros((1, d), dtype=float)
    x_whatif = np.zeros((1, d), dtype=float)
    y_whatif = np.array([[100.0]], dtype=float)
    params = ENNParams(
        k_num_neighbors=1, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    post_incl = model.conditional_posterior(
        x_whatif,
        y_whatif,
        x_test,
        params=params,
        flags=PosteriorFlags(exclude_nearest=False, observation_noise=True),
    )
    post_excl = model.conditional_posterior(
        x_whatif,
        y_whatif,
        x_test,
        params=params,
        flags=PosteriorFlags(exclude_nearest=True, observation_noise=True),
    )
    assert np.allclose(post_incl.mu, 100.0)
    assert not np.allclose(post_excl.mu, 100.0)


def test_conditional_posterior_matches_augmented_model_exactly():
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
    enn_a = EpistemicNearestNeighbors(train_x, train_y, train_yvar=None)
    post_a = enn_a.conditional_posterior(
        x_whatif, y_whatif, x_test, params=params, flags=flags
    )
    enn_b = EpistemicNearestNeighbors(
        np.concatenate([train_x, x_whatif], axis=0),
        np.concatenate([train_y, y_whatif], axis=0),
        train_yvar=None,
    )
    post_b = enn_b.posterior(x_test, params=params, flags=flags)
    assert np.allclose(post_a.mu, post_b.mu)
    assert np.allclose(post_a.se, post_b.se)
