from __future__ import annotations
import numpy as np
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


def test_posterior_function_sample_basic():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(train_x, train_y, 0.1 * np.ones_like(train_y))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    sample = model.posterior_function_draw(x_test, params, function_seeds=[123])[0]
    assert sample.shape == (5, 1) and np.all(np.isfinite(sample))


def test_posterior_function_sample_deterministic():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    model = EpistemicNearestNeighbors(train_x, train_x.sum(axis=1, keepdims=True))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    sample1 = model.posterior_function_draw(x_test, params, function_seeds=[42])[0]
    assert np.allclose(
        sample1, model.posterior_function_draw(x_test, params, function_seeds=[42])[0]
    )
    assert not np.allclose(
        sample1, model.posterior_function_draw(x_test, params, function_seeds=[43])[0]
    )


def test_posterior_function_sample_batch_basic():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(train_x, train_y, 0.1 * np.ones_like(train_y))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    samples = model.posterior_function_draw(x_test, params, function_seeds=[10, 20, 30])
    assert samples.shape == (3, 5, 1) and np.all(np.isfinite(samples))


def test_posterior_function_sample_batch_matches_single_seed():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    model = EpistemicNearestNeighbors(train_x, train_x.sum(axis=1, keepdims=True))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    batch = model.posterior_function_draw(
        x_test, params, function_seeds=[100, 200, 300]
    )
    for i, seed in enumerate([100, 200, 300]):
        assert np.allclose(
            batch[i],
            model.posterior_function_draw(x_test, params, function_seeds=[seed])[0],
        )


def test_posterior_function_sample_batch_with_multiple_metrics():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    model = EpistemicNearestNeighbors(train_x, rng.standard_normal((20, 2)))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    samples = model.posterior_function_draw(x_test, params, function_seeds=[1, 2, 3, 4])
    assert samples.shape == (4, 5, 2) and np.all(np.isfinite(samples))


def test_posterior_function_sample_batch_empty_k():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((2, 3))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(train_x, train_y)
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    samples = model.posterior_function_draw(
        x_test,
        params,
        function_seeds=[1, 2],
        flags=PosteriorFlags(exclude_nearest=True),
    )
    assert samples.shape == (2, 5, 1)


def test_posterior_function_sample_with_observation_noise():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(train_x, train_y, 0.5 * np.ones_like(train_y))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    sample_no_noise = model.posterior_function_draw(
        x_test, params, function_seeds=[42]
    )[0]
    sample_with_noise = model.posterior_function_draw(
        x_test,
        params,
        function_seeds=[42],
        flags=PosteriorFlags(observation_noise=True),
    )[0]
    assert sample_no_noise.shape == sample_with_noise.shape == (5, 1)
    assert not np.allclose(sample_no_noise, sample_with_noise)
