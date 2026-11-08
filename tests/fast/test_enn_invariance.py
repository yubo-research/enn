from __future__ import annotations

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from evals.harness.work_dirs import fresh_work_dir


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
    model_base = EpistemicNearestNeighbors(
        train_x, train_y, train_yvar, work_dir=fresh_work_dir()
    )
    model_scaled = EpistemicNearestNeighbors(
        train_x,
        train_y * 100.0,
        train_yvar * 10000.0,
        work_dir=fresh_work_dir(),
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
    model_base = EpistemicNearestNeighbors(
        train_x, train_y, train_yvar, work_dir=fresh_work_dir()
    )
    model_shifted = EpistemicNearestNeighbors(
        train_x, train_y + 1000.0, train_yvar, work_dir=fresh_work_dir()
    )
    x_test, params = rng.standard_normal((10, 3)), _params(5)
    post_base = model_base.posterior(x_test, params=params)
    post_shifted = model_shifted.posterior(x_test, params=params)
    assert np.allclose(post_shifted.mu, post_base.mu + 1000.0, rtol=1e-10)
    assert np.allclose(post_shifted.se, post_base.se, rtol=1e-10)
