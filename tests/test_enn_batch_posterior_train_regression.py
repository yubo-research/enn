"""Regression: batch_posterior must agree with posterior when x is training data."""

from __future__ import annotations

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


def _user_reported_train_data(rng: np.random.Generator):
    """x ~ N(0,1), y ~ N(0, 100^2), flat index, n=20, d=1."""
    n, d = 20, 1
    train_x = rng.standard_normal((n, d))
    train_y = rng.normal(0.0, 100.0, size=(n, d))
    return train_x, train_y


def test_posterior_full_train_matches_single_row_user_reported_params():
    """posterior(x_train)[i] must match posterior(x_train[i:i+1]) for reported scenario."""
    rng = np.random.default_rng(42)
    train_x, train_y = _user_reported_train_data(rng)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar=None)
    params = ENNParams(
        10,
        epistemic_variance_scale=80.0,
        aleatoric_variance_scale=0.0,
    )
    flags = PosteriorFlags()
    post_full = model.posterior(train_x, params=params, flags=flags)
    for i in range(train_x.shape[0]):
        post_i = model.posterior(train_x[i : i + 1], params=params, flags=flags)
        np.testing.assert_allclose(post_full.mu[i], post_i.mu[0], rtol=0, atol=0)
        np.testing.assert_allclose(post_full.se[i], post_i.se[0], rtol=0, atol=0)


def test_batch_posterior_matches_posterior_user_reported_params():
    """k=10, n=20, d=1, epistemic=80, aleatoric=0 on x_train (reported mismatch scenario)."""
    rng = np.random.default_rng(42)
    train_x, train_y = _user_reported_train_data(rng)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar=None)
    params = ENNParams(
        10,
        epistemic_variance_scale=80.0,
        aleatoric_variance_scale=0.0,
    )
    flags = PosteriorFlags()
    post_batch = model.batch_posterior(train_x, [params], flags=flags)
    post = model.posterior(train_x, params=params, flags=flags)
    np.testing.assert_allclose(post_batch.mu[0], post.mu, rtol=0, atol=0)
    np.testing.assert_allclose(post_batch.se[0], post.se, rtol=0, atol=0)
    for i in range(train_x.shape[0]):
        post_i = model.posterior(train_x[i : i + 1], params=params, flags=flags)
        np.testing.assert_allclose(post_batch.mu[0, i], post_i.mu[0], rtol=0, atol=0)
        np.testing.assert_allclose(post_batch.se[0, i], post_i.se[0], rtol=0, atol=0)


def test_batch_posterior_matches_posterior_on_train_x():
    """Shared-neighbor batch path on x_train must match row-wise posterior (fit LOO flags)."""
    rng = np.random.default_rng(17)
    n, d = 12, 3
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    paramss = [
        ENNParams(3, epistemic_variance_scale=0.5, aleatoric_variance_scale=0.1),
        ENNParams(3, epistemic_variance_scale=2.0, aleatoric_variance_scale=0.0),
    ]
    flags = PosteriorFlags(exclude_nearest=True, observation_noise=True)
    post_batch = model.batch_posterior(train_x, paramss, flags=flags)
    for i, params in enumerate(paramss):
        post = model.posterior(train_x, params=params, flags=flags)
        np.testing.assert_allclose(post_batch.mu[i], post.mu, rtol=0, atol=0)
        np.testing.assert_allclose(post_batch.se[i], post.se, rtol=0, atol=0)
