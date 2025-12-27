from __future__ import annotations

import numpy as np
import pytest

from enn.enn import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


def _params(
    k: int, *, epi_var_scale: float = 1.0, ale_homoscedastic_scale: float = 0.0
):
    return ENNParams(
        k=int(k),
        epi_var_scale=float(epi_var_scale),
        ale_homoscedastic_scale=float(ale_homoscedastic_scale),
    )


def test_ennnormal_sample_shape_and_clip():
    from enn.enn.enn_normal import ENNNormal

    rng = np.random.default_rng(0)
    mu = np.array([[0.0, 1.0]], dtype=float)
    se = np.array([[1.0, 2.0]], dtype=float)
    normal = ENNNormal(mu=mu, se=se)
    samples = normal.sample(5, clip=1.0, rng=rng)
    assert samples.shape == (1, 2, 5)
    assert np.all(samples >= mu.min() - 2.0)
    assert np.all(samples <= mu.max() + 2.0)


def test_epistemic_nearest_neighbors_posterior_and_var_scale():
    import conftest

    model, _train_x, _train_y, _train_yvar, rng = conftest.make_enn_model()
    x_test = rng.standard_normal((4, 3))
    params = ENNParams(k=3, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    post = model.posterior(x_test, params=params)
    assert post.mu.shape == (4, 1)
    assert post.se.shape == (4, 1)
    post_changed = model.posterior(
        x_test,
        params=ENNParams(k=5, epi_var_scale=0.5, ale_homoscedastic_scale=0.0),
        flags=PosteriorFlags(exclude_nearest=True),
    )
    assert post_changed.mu.shape == (4, 1)
    assert post_changed.se.shape == (4, 1)


def test_epistemic_nearest_neighbors_with_no_observations_returns_prior_like_posterior():
    rng = np.random.default_rng(0)
    d = 3
    x = np.zeros((0, d), dtype=float)
    y = np.zeros((0, 1), dtype=float)
    yvar = np.ones_like(y, dtype=float)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((5, d))
    post = model.posterior(
        x_test,
        params=ENNParams(k=3, epi_var_scale=1.0, ale_homoscedastic_scale=0.0),
    )
    assert post.mu.shape == (5, 1)
    assert post.se.shape == (5, 1)
    assert np.allclose(post.mu, 0.0)
    assert np.allclose(post.se, 1.0)


@pytest.mark.parametrize("num_obs", [1, 2, 3])
def test_epistemic_nearest_neighbors_with_few_observations_has_valid_posterior(
    num_obs: int,
):
    rng = np.random.default_rng(0)
    d = 3
    x = rng.standard_normal((num_obs, d))
    y = (x.sum(axis=1, keepdims=True)).astype(float)
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((5, d))
    post = model.posterior(x_test, params=_params(3))
    assert post.mu.shape == (5, 1)
    assert post.se.shape == (5, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


@pytest.mark.parametrize(
    "flags,k_vals",
    [
        (PosteriorFlags(), [3, 5, 7]),
        (PosteriorFlags(exclude_nearest=True), [3, 5]),
    ],
)
def test_batch_posterior_matches_individual_posterior_calls(flags, k_vals):
    import conftest

    model, _train_x, _train_y, _train_yvar, rng = conftest.make_enn_model()
    x_test = rng.standard_normal((4, 3))
    paramss = [
        ENNParams(k=k, epi_var_scale=1.0 / (i + 1), ale_homoscedastic_scale=0.0)
        for i, k in enumerate(k_vals)
    ]
    post_batch = model.batch_posterior(x_test, paramss, flags=flags)
    assert post_batch.mu.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    assert post_batch.se.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    for i, params in enumerate(paramss):
        post = model.posterior(x_test, params=params, flags=flags)
        assert np.allclose(post_batch.mu[i], post.mu) and np.allclose(
            post_batch.se[i], post.se
        )


def test_epistemic_nearest_neighbors_with_sobol_indices():
    rng = np.random.default_rng(0)
    n = 50
    d = 3
    x = rng.standard_normal((n, d))
    y = (x[:, 0] + 0.1 * x[:, 1] + 0.01 * rng.standard_normal(n)).reshape(-1, 1)
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((4, d))
    params = ENNParams(k=3, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    post = model.posterior(x_test, params=params)
    assert post.mu.shape == (4, 1) and post.se.shape == (4, 1)
    assert np.all(np.isfinite(post.mu)) and np.all(np.isfinite(post.se))


def test_epistemic_nearest_neighbors_multiple_metrics():
    rng = np.random.default_rng(0)
    n = 20
    d = 3
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 2))
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((4, d))
    params = ENNParams(k=3, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    post = model.posterior(x_test, params=params)
    assert post.mu.shape == (4, 2) and post.se.shape == (4, 2)


def test_batch_posterior_exclude_nearest_with_k_larger_than_available():
    """
    Forces the off-by-one bug when exclude_nearest=True and k > len(self) - 1.

    With len(self)=5, max_k=10, exclude_nearest=True:
    - search_k = min(11, 5) = 5
    - After slicing [:, 1:], arrays have 4 columns
    - BUG: k = min(10, 5) = 5, but should be min(10, 4) = 4
    """
    rng = np.random.default_rng(0)
    n = 5
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y = (train_x.sum(axis=1, keepdims=True)).astype(float)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)

    x_test = rng.standard_normal((4, d))
    params = ENNParams(k=10, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    post = model.batch_posterior(
        x_test, [params], flags=PosteriorFlags(exclude_nearest=True)
    )
    assert post.mu.shape == (1, 4, 1) and post.se.shape == (1, 4, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


def test_epistemic_nearest_neighbors_with_yvar_none():
    rng = np.random.default_rng(42)
    n = 20
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True) + rng.standard_normal((n, 1)) * 0.1

    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar=None)

    assert len(model) == n
    assert model.train_yvar is None

    x_test = rng.standard_normal((10, d))
    params = ENNParams(k=5, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    post = model.posterior(x_test, params=params)

    assert post.mu.shape == (10, 1)
    assert post.se.shape == (10, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


def test_epistemic_nearest_neighbors_constant_y_scale_is_safe():
    rng = np.random.default_rng(0)
    n = 20
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y = np.zeros((n, 1), dtype=float)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)

    x_test = rng.standard_normal((5, d))
    params = ENNParams(k=5, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    post = model.posterior(x_test, params=params)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


def test_epistemic_nearest_neighbors_init_validates_inputs():
    rng = np.random.default_rng(0)

    # x must be 2D
    with pytest.raises(ValueError):
        EpistemicNearestNeighbors(rng.random(10), np.zeros((10, 1)))

    # y must be 2D
    with pytest.raises(ValueError):
        EpistemicNearestNeighbors(rng.random((10, 3)), rng.random(10))

    # x and y must have matching rows
    with pytest.raises(ValueError):
        EpistemicNearestNeighbors(rng.random((10, 3)), rng.random((5, 1)))

    # yvar must be 2D if provided
    with pytest.raises(ValueError):
        EpistemicNearestNeighbors(
            rng.random((10, 3)), rng.random((10, 1)), rng.random(10)
        )

    # yvar must match y shape
    with pytest.raises(ValueError):
        EpistemicNearestNeighbors(
            rng.random((10, 3)), rng.random((10, 1)), rng.random((10, 2))
        )


def test_epistemic_nearest_neighbors_init_explicit():
    rng = np.random.default_rng(42)
    n, d = 20, 3
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True)
    train_yvar = 0.1 * np.ones_like(train_y)

    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    assert len(model) == n
    assert model.num_outputs == 1
    assert model.train_x is not None
    assert model.train_y is not None
    assert model.train_yvar is not None
