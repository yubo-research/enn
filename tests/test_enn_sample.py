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
    draws, idx = model.posterior_function_draw(x_test, params, function_seeds=[123])
    sample = draws[:, :, 0]
    assert sample.shape == (5, 1) and np.all(np.isfinite(sample))
    assert idx.shape == (5, 5)


def test_posterior_function_sample_deterministic():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    model = EpistemicNearestNeighbors(train_x, train_x.sum(axis=1, keepdims=True))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    sample1 = model.posterior_function_draw(x_test, params, function_seeds=[42])[0][
        :, :, 0
    ]
    assert np.allclose(
        sample1,
        model.posterior_function_draw(x_test, params, function_seeds=[42])[0][:, :, 0],
    )
    assert not np.allclose(
        sample1,
        model.posterior_function_draw(x_test, params, function_seeds=[43])[0][:, :, 0],
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
    samples, idx = model.posterior_function_draw(
        x_test, params, function_seeds=[10, 20, 30]
    )
    assert samples.shape == (5, 1, 3) and np.all(np.isfinite(samples))
    assert idx.shape == (5, 5)


def test_posterior_function_sample_batch_matches_single_seed():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    model = EpistemicNearestNeighbors(train_x, train_x.sum(axis=1, keepdims=True))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    batch, _ = model.posterior_function_draw(
        x_test, params, function_seeds=[100, 200, 300]
    )
    for i, seed in enumerate([100, 200, 300]):
        assert np.allclose(
            batch[:, :, i],
            model.posterior_function_draw(x_test, params, function_seeds=[seed])[0][
                :, :, 0
            ],
        )


def test_posterior_function_sample_batch_with_multiple_metrics():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    model = EpistemicNearestNeighbors(train_x, rng.standard_normal((20, 2)))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    samples, _ = model.posterior_function_draw(
        x_test, params, function_seeds=[1, 2, 3, 4]
    )
    assert samples.shape == (5, 2, 4) and np.all(np.isfinite(samples))


def test_posterior_function_sample_batch_empty_k():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((2, 3))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(train_x, train_y)
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    samples, idx = model.posterior_function_draw(
        x_test,
        params,
        function_seeds=[1, 2],
        flags=PosteriorFlags(exclude_nearest=True),
    )
    # Novel queries keep all available neighbors under exclude (n=2 → 2 cols).
    assert samples.shape == (5, 1, 2)
    assert idx.shape == (5, 2)


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
    )[0][:, :, 0]
    sample_with_noise = model.posterior_function_draw(
        x_test,
        params,
        function_seeds=[42],
        flags=PosteriorFlags(observation_noise=True),
    )[0][:, :, 0]
    assert sample_no_noise.shape == sample_with_noise.shape == (5, 1)
    assert not np.allclose(sample_no_noise, sample_with_noise)


def test_posterior_function_draw_aleatoric_is_independent_of_noise_field():
    """Aleatoric noise must not be folded into the correlated neighbor field.

    With observation_noise=True, marginal draw std ≈ total se, but pairwise
    correlation across queries must drop vs the epistemic-only draw.
    """
    rng = np.random.default_rng(0)
    train_x = rng.standard_normal((40, 2))
    train_y = train_x.sum(axis=1, keepdims=True)
    train_yvar = 0.25 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar, scale_x=False)
    x_test = rng.standard_normal((6, 2))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    flags_off = PosteriorFlags(observation_noise=False)
    flags_on = PosteriorFlags(observation_noise=True)
    seeds = list(range(1500))

    post_on = model.posterior(x_test, params=params, flags=flags_on)
    draws_off = model.posterior_function_draw(
        x_test, params, function_seeds=seeds, flags=flags_off
    )[0][:, 0, :]
    draws_on = model.posterior_function_draw(
        x_test, params, function_seeds=seeds, flags=flags_on
    )[0][:, 0, :]

    np.testing.assert_allclose(
        draws_on.std(axis=1, ddof=1), post_on.se[:, 0], rtol=0.08, atol=0.05
    )

    # Epistemic-only field is shared; independent aleatoric must reduce corr.
    corr_off = np.corrcoef(draws_off[0], draws_off[1])[0, 1]
    corr_on = np.corrcoef(draws_on[0], draws_on[1])[0, 1]
    se_ratio = (post_on.se_epi[0, 0] * post_on.se_epi[1, 0]) / (
        post_on.se[0, 0] * post_on.se[1, 0]
    )
    assert corr_on < corr_off - 1e-3
    assert abs(corr_on - corr_off * se_ratio) < 0.08
