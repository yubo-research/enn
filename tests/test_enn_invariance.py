from __future__ import annotations
import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams


def _params(
    k: int,
    *,
    epistemic_variance_scale: float = 1.0,
    aleatoric_variance_scale: float = 0.0,
):
    return ENNParams(
        k_num_neighbors=int(k),
        epistemic_variance_scale=float(epistemic_variance_scale),
        aleatoric_variance_scale=float(aleatoric_variance_scale),
    )


def _make_single_metric_train_data(*, rng, n: int, d: int, noise_std: float):
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True) + rng.standard_normal((n, 1)) * float(
        noise_std
    )
    return train_x, train_y, 0.1 * np.ones_like(train_y)


def test_epistemic_nearest_neighbors_scale_invariance():
    rng = np.random.default_rng(42)
    train_x, train_y, train_yvar = _make_single_metric_train_data(
        rng=rng, n=20, d=3, noise_std=0.1
    )
    model_base = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    model_scaled = EpistemicNearestNeighbors(
        train_x, train_y * 100.0, train_yvar * 10000.0
    )
    x_test, params = rng.standard_normal((10, 3)), _params(5)
    post_base = model_base.posterior(x_test, params=params)
    post_scaled = model_scaled.posterior(x_test, params=params)
    assert np.allclose(post_scaled.mu, post_base.mu * 100.0, rtol=1e-10)
    assert np.allclose(post_scaled.se, post_base.se * 100.0, rtol=1e-10)


def test_epistemic_nearest_neighbors_shift_invariance():
    rng = np.random.default_rng(42)
    train_x, train_y, train_yvar = _make_single_metric_train_data(
        rng=rng, n=20, d=3, noise_std=0.1
    )
    model_base = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    model_shifted = EpistemicNearestNeighbors(train_x, train_y + 1000.0, train_yvar)
    x_test, params = rng.standard_normal((10, 3)), _params(5)
    post_base = model_base.posterior(x_test, params=params)
    post_shifted = model_shifted.posterior(x_test, params=params)
    assert np.allclose(post_shifted.mu, post_base.mu + 1000.0, rtol=1e-10)
    assert np.allclose(post_shifted.se, post_base.se, rtol=1e-10)


def test_epistemic_nearest_neighbors_x_rescaling_is_invariant_when_scale_x_enabled():
    rng = np.random.default_rng(0)
    train_x = rng.standard_normal((50, 4))
    train_y = train_x.sum(axis=1, keepdims=True)
    scale = np.array([[100.0, 0.1, 3.0, 1.0]])
    x_test = rng.standard_normal((10, 4))
    params = ENNParams(
        k_num_neighbors=7, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    model = EpistemicNearestNeighbors(
        train_x, train_y, 0.1 * np.ones_like(train_y), scale_x=True
    )
    model_scaled = EpistemicNearestNeighbors(
        train_x * scale, train_y, 0.1 * np.ones_like(train_y), scale_x=True
    )
    post = model.posterior(x_test, params=params)
    post_scaled = model_scaled.posterior(x_test * scale, params=params)
    assert np.allclose(post.mu, post_scaled.mu, rtol=1e-6, atol=1e-8)
    assert np.allclose(post.se, post_scaled.se, rtol=1e-6, atol=1e-8)
