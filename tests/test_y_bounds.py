"""Tests for optional per-metric y bounds (output warp)."""

from __future__ import annotations

import numpy as np
import pytest

from enn import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from enn.turbo.config import ENNFitConfig, ENNSurrogateConfig, turbo_enn_config
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.rust_optimizer import create_optimizer


def test_public_posterior_and_train_y_natural_logit():
    bounds = np.array([[0.0, 1.0]], dtype=float)
    train_x = np.array([[0.0], [1.0]], dtype=float)
    train_y = np.array([[0.1], [0.9]], dtype=float)
    model = EpistemicNearestNeighbors(train_x, train_y, y_bounds=bounds)

    np.testing.assert_allclose(model._train_y, train_y)

    from enn.enn_rust import model as enn_rust_model

    _, y_z, _ = enn_rust_model.train_rows_at_warped(model._rust_model, [0, 1])
    y_z = np.asarray(y_z, dtype=float)
    expected_z = np.log(train_y / (1.0 - train_y))
    np.testing.assert_allclose(y_z, expected_z)
    assert not np.allclose(y_z, train_y)

    params = ENNParams(
        k_num_neighbors=1,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )
    post = model.posterior(train_x, params=params)
    assert np.all(post.mu > 0.0) and np.all(post.mu < 1.0)

    rng = np.random.default_rng(0)
    draws = post.sample(32, rng)
    assert draws.shape == (*post.mu.shape, 32)
    assert np.all(draws > 0.0) and np.all(draws < 1.0)


def test_oob_rejected():
    bounds = np.array([[0.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="strictly"):
        EpistemicNearestNeighbors(
            np.array([[0.0]], dtype=float),
            np.array([[0.0]], dtype=float),
            y_bounds=bounds,
        )


def test_shape_strict_no_broadcast():
    bounds = np.array([[0.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="num_metrics"):
        EpistemicNearestNeighbors(
            np.array([[0.0], [1.0]], dtype=float),
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
            y_bounds=bounds,
        )


def test_unbounded_default_matches_identity():
    train_x = np.array([[0.0], [1.0]], dtype=float)
    train_y = np.array([[1.0], [2.0]], dtype=float)
    model = EpistemicNearestNeighbors(train_x, train_y)
    params = ENNParams(
        k_num_neighbors=1,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )
    post = model.posterior(train_x, params=params)
    rng = np.random.default_rng(1)
    draws = post.sample(8, rng)
    assert np.isfinite(draws).all()


def test_optimizer_bounded_open_unit_interval():
    bounds = np.array([[0.0, 1.0]], dtype=float)
    cfg = turbo_enn_config(
        enn=ENNSurrogateConfig(
            k=2,
            fit=ENNFitConfig(num_fit_samples=5),
            y_bounds=bounds,
        ),
        num_init=2,
    )
    rng = np.random.default_rng(0)
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0]], dtype=float),
        config=cfg,
        rng=rng,
    )
    x = opt.ask(2)
    y = np.array([[0.2], [0.8]], dtype=float)
    opt.tell(x, y)
    y_obs = opt._y_obs.view()
    assert y_obs.shape[0] >= 2
    assert np.all(y_obs > 0.0) and np.all(y_obs < 1.0)
    x2 = opt.ask(1)
    assert x2.shape == (1, 1)


def test_disk_reopen_loads_y_bounds(tmp_path):
    bounds = np.array([[0.0, 1.0]], dtype=float)
    work = tmp_path / "enn_yb"
    train_x = np.array([[0.0], [1.0]], dtype=float)
    train_y = np.array([[0.2], [0.8]], dtype=float)
    model = EpistemicNearestNeighbors(
        train_x,
        train_y,
        y_bounds=bounds,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work),
        enn_storage="disk",
    )
    model.persist_index_to_disk()

    reopened = EpistemicNearestNeighbors(
        np.zeros((0, 1)),
        np.zeros((0, 1)),
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work),
        enn_storage="disk",
    )
    np.testing.assert_allclose(reopened._train_y, train_y)
    with pytest.raises(ValueError, match="do not match"):
        EpistemicNearestNeighbors(
            np.zeros((0, 1)),
            np.zeros((0, 1)),
            y_bounds=np.array([[0.0, 2.0]], dtype=float),
            index_driver=ENNIndexDriver.BPANN_DISK,
            work_dir=str(work),
            enn_storage="disk",
        )


def test_multi_metric_bounds():
    bounds = np.array([[0.0, 1.0], [0.0, np.inf]], dtype=float)
    train_x = np.array([[0.0], [1.0]], dtype=float)
    train_y = np.array([[0.2, 1.5], [0.8, 3.0]], dtype=float)
    model = EpistemicNearestNeighbors(train_x, train_y, y_bounds=bounds)
    np.testing.assert_allclose(model._train_y, train_y)
    params = ENNParams(
        k_num_neighbors=1,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )
    post = model.posterior(train_x, params=params)
    assert post.mu.shape[-1] == 2
    assert np.all(post.mu[..., 0] > 0.0) and np.all(post.mu[..., 0] < 1.0)
    assert np.all(post.mu[..., 1] > 0.0)
