from __future__ import annotations

import pytest


def test_ennnormal_sample_shape_and_clip():
    import numpy as np

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

    from enn.enn.enn_params import ENNParams

    model, _train_x, _train_y, _train_yvar, rng = conftest.make_enn_model()
    x_test = rng.standard_normal((4, 3))
    params = ENNParams(k=3, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    post = model.posterior(x_test, params=params, exclude_nearest=False)
    assert post.mu.shape == (4, 1)
    assert post.se.shape == (4, 1)
    post_changed = model.posterior(
        x_test,
        params=ENNParams(k=5, epi_var_scale=0.5, ale_homoscedastic_scale=0.0),
        exclude_nearest=True,
    )
    assert post_changed.mu.shape == (4, 1)
    assert post_changed.se.shape == (4, 1)


def test_epistemic_nearest_neighbors_with_no_observations_returns_prior_like_posterior():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

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
        exclude_nearest=False,
    )
    assert post.mu.shape == (5, 1)
    assert post.se.shape == (5, 1)
    assert np.allclose(post.mu, 0.0)
    assert np.allclose(post.se, 1.0)


@pytest.mark.parametrize("num_obs", [1, 2, 3])
def test_epistemic_nearest_neighbors_with_few_observations_has_valid_posterior(
    num_obs: int,
):
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    d = 3
    x = rng.standard_normal((num_obs, d))
    y = (x.sum(axis=1, keepdims=True)).astype(float)
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((5, d))
    post = model.posterior(
        x_test,
        params=ENNParams(k=3, epi_var_scale=1.0, ale_homoscedastic_scale=0.0),
        exclude_nearest=False,
    )
    assert post.mu.shape == (5, 1)
    assert post.se.shape == (5, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


def test_batch_posterior_matches_individual_posterior_calls():
    import conftest
    import numpy as np

    from enn.enn.enn_params import ENNParams

    model, _train_x, _train_y, _train_yvar, rng = conftest.make_enn_model()
    x_test = rng.standard_normal((4, 3))
    paramss = [
        ENNParams(k=3, epi_var_scale=1.0, ale_homoscedastic_scale=0.0),
        ENNParams(k=5, epi_var_scale=0.5, ale_homoscedastic_scale=0.0),
        ENNParams(k=7, epi_var_scale=2.0, ale_homoscedastic_scale=0.0),
    ]
    post_batch = model.batch_posterior(x_test, paramss, exclude_nearest=False)
    assert post_batch.mu.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    assert post_batch.se.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    for i, params in enumerate(paramss):
        post = model.posterior(x_test, params=params, exclude_nearest=False)
        assert np.allclose(post_batch.mu[i], post.mu)
        assert np.allclose(post_batch.se[i], post.se)


def test_batch_posterior_matches_individual_posterior_calls_with_exclude_nearest():
    import conftest
    import numpy as np

    from enn.enn.enn_params import ENNParams

    model, _train_x, _train_y, _train_yvar, rng = conftest.make_enn_model()
    x_test = rng.standard_normal((4, 3))
    paramss = [
        ENNParams(k=3, epi_var_scale=1.0, ale_homoscedastic_scale=0.0),
        ENNParams(k=5, epi_var_scale=0.5, ale_homoscedastic_scale=0.0),
    ]
    post_batch = model.batch_posterior(x_test, paramss, exclude_nearest=True)
    assert post_batch.mu.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    assert post_batch.se.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    for i, params in enumerate(paramss):
        post = model.posterior(x_test, params=params, exclude_nearest=True)
        assert np.allclose(post_batch.mu[i], post.mu)
        assert np.allclose(post_batch.se[i], post.se)


def test_epistemic_nearest_neighbors_with_sobol_indices():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    n = 50
    d = 3
    x = rng.standard_normal((n, d))
    y = (x[:, 0] + 0.1 * x[:, 1] + 0.01 * rng.standard_normal(n)).reshape(-1, 1)
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((4, d))
    params = ENNParams(k=3, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    post = model.posterior(x_test, params=params, exclude_nearest=False)
    assert post.mu.shape == (4, 1)
    assert post.se.shape == (4, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


def test_epistemic_nearest_neighbors_multiple_metrics():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    n = 20
    d = 3
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 2))
    yvar = 0.1 * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    x_test = rng.standard_normal((4, d))
    params = ENNParams(k=3, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    post = model.posterior(x_test, params=params, exclude_nearest=False)
    assert post.mu.shape == (4, 2)
    assert post.se.shape == (4, 2)


def test_neighbors_returns_correct_number_and_ordering():
    import conftest
    import numpy as np

    model, train_x, train_y, _train_yvar, _rng = conftest.make_enn_model()
    d = 3

    # Query point at origin
    x_query = np.zeros(d, dtype=float)
    neighbors = model.neighbors(x_query, k=5, exclude_nearest=False)

    assert len(neighbors) == 5
    neighbor_indices = []
    for x_neighbor, y_neighbor in neighbors:
        assert x_neighbor.shape == (d,)
        assert y_neighbor.shape == (1,)
        # Find which training point this corresponds to
        row_diffs = np.abs(train_x - x_neighbor[np.newaxis, :])
        matches = np.all(row_diffs < 1e-6, axis=1)
        assert np.any(matches), "Neighbor x not found in training data"
        idx = np.where(matches)[0][0]
        neighbor_indices.append(idx)
        assert np.allclose(
            train_y[idx], y_neighbor
        ), "Neighbor y doesn't match training data"

    # Verify ordering: distances should be non-decreasing
    distances = [np.linalg.norm(train_x[idx] - x_query) for idx in neighbor_indices]
    for i in range(len(distances) - 1):
        assert (
            distances[i] <= distances[i + 1] + 1e-6
        ), f"Distances not ordered: {distances}"


def test_neighbors_exclude_nearest():
    import conftest
    import numpy as np

    model, train_x, _train_y, _train_yvar, _rng = conftest.make_enn_model()

    # Query point exactly matching a training point
    x_query = train_x[5].copy()
    neighbors_exclude = model.neighbors(x_query, k=5, exclude_nearest=True)
    neighbors_include = model.neighbors(x_query, k=5, exclude_nearest=False)

    assert len(neighbors_exclude) == 5
    assert len(neighbors_include) == 5
    # The first neighbor when include=True should be the exact match
    assert np.allclose(neighbors_include[0][0], x_query)
    # The first neighbor when exclude=True should NOT be the exact match
    assert not np.allclose(neighbors_exclude[0][0], x_query)


def test_neighbors_with_empty_observations():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors

    d = 3
    train_x = np.zeros((0, d), dtype=float)
    train_y = np.zeros((0, 1), dtype=float)
    train_yvar = np.ones((0, 1), dtype=float)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)

    x_query = np.zeros(d, dtype=float)
    neighbors = model.neighbors(x_query, k=5, exclude_nearest=False)
    assert neighbors == []


def test_neighbors_k_larger_than_available():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors

    rng = np.random.default_rng(0)
    n = 5
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y = (train_x.sum(axis=1, keepdims=True)).astype(float)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)

    x_query = np.zeros(d, dtype=float)
    neighbors = model.neighbors(x_query, k=20, exclude_nearest=False)
    assert len(neighbors) == n


def test_neighbors_k_zero():
    import conftest
    import numpy as np

    model, _train_x, _train_y, _train_yvar, _rng = conftest.make_enn_model(n=10)

    x_query = np.zeros(3, dtype=float)
    neighbors = model.neighbors(x_query, k=0, exclude_nearest=False)
    assert neighbors == []


def test_neighbors_with_multiple_metrics():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors

    rng = np.random.default_rng(0)
    n = 15
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y = rng.standard_normal((n, 2))
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)

    x_query = np.zeros(d, dtype=float)
    neighbors = model.neighbors(x_query, k=5, exclude_nearest=False)

    assert len(neighbors) == 5
    for x_neighbor, y_neighbor in neighbors:
        assert x_neighbor.shape == (d,)
        assert y_neighbor.shape == (2,)


def test_neighbors_accepts_2d_input():
    import conftest
    import numpy as np

    model, _train_x, _train_y, _train_yvar, _rng = conftest.make_enn_model(n=10)
    d = 3

    # Test with 1D input
    x_query_1d = np.zeros(d, dtype=float)
    neighbors_1d = model.neighbors(x_query_1d, k=3, exclude_nearest=False)

    # Test with 2D input
    x_query_2d = np.zeros((1, d), dtype=float)
    neighbors_2d = model.neighbors(x_query_2d, k=3, exclude_nearest=False)

    assert len(neighbors_1d) == len(neighbors_2d) == 3
    for (x1, y1), (x2, y2) in zip(neighbors_1d, neighbors_2d):
        assert np.allclose(x1, x2)
        assert np.allclose(y1, y2)


def test_neighbors_exclude_nearest_requires_multiple_observations():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors

    rng = np.random.default_rng(0)
    d = 3
    train_x = rng.standard_normal((1, d))
    train_y = np.array([[1.0]])
    train_yvar = np.array([[0.1]])
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)

    x_query = np.zeros(d, dtype=float)
    with pytest.raises(
        ValueError, match="exclude_nearest=True requires at least 2 observations"
    ):
        model.neighbors(x_query, k=1, exclude_nearest=True)


def test_batch_posterior_exclude_nearest_with_k_larger_than_available():
    """
    Forces the off-by-one bug when exclude_nearest=True and k > len(self) - 1.

    With len(self)=5, max_k=10, exclude_nearest=True:
    - search_k = min(11, 5) = 5
    - After slicing [:, 1:], arrays have 4 columns
    - BUG: k = min(10, 5) = 5, but should be min(10, 4) = 4
    """
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    n = 5
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y = (train_x.sum(axis=1, keepdims=True)).astype(float)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)

    x_test = rng.standard_normal((4, d))
    params = ENNParams(k=10, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    post = model.batch_posterior(x_test, [params], exclude_nearest=True)
    assert post.mu.shape == (1, 4, 1)
    assert post.se.shape == (1, 4, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


def test_epistemic_nearest_neighbors_scale_invariance():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(42)
    n = 20
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y_base = (
        train_x.sum(axis=1, keepdims=True) + rng.standard_normal((n, 1)) * 0.1
    )
    train_yvar_base = 0.1 * np.ones_like(train_y_base)

    scale_factor = 100.0
    train_y_scaled = train_y_base * scale_factor
    train_yvar_scaled = train_yvar_base * (scale_factor**2)

    model_base = EpistemicNearestNeighbors(train_x, train_y_base, train_yvar_base)
    model_scaled = EpistemicNearestNeighbors(train_x, train_y_scaled, train_yvar_scaled)

    x_test = rng.standard_normal((10, d))
    params = ENNParams(k=5, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)

    post_base = model_base.posterior(x_test, params=params)
    post_scaled = model_scaled.posterior(x_test, params=params)

    assert np.allclose(post_scaled.mu, post_base.mu * scale_factor, rtol=1e-10)
    assert np.allclose(post_scaled.se, post_base.se * scale_factor, rtol=1e-10)


def test_epistemic_nearest_neighbors_shift_invariance():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(42)
    n = 20
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y_base = (
        train_x.sum(axis=1, keepdims=True) + rng.standard_normal((n, 1)) * 0.1
    )
    train_yvar = 0.1 * np.ones_like(train_y_base)

    shift = 1000.0
    train_y_shifted = train_y_base + shift

    model_base = EpistemicNearestNeighbors(train_x, train_y_base, train_yvar)
    model_shifted = EpistemicNearestNeighbors(train_x, train_y_shifted, train_yvar)

    x_test = rng.standard_normal((10, d))
    params = ENNParams(k=5, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)

    post_base = model_base.posterior(x_test, params=params)
    post_shifted = model_shifted.posterior(x_test, params=params)

    assert np.allclose(post_shifted.mu, post_base.mu + shift, rtol=1e-10)
    assert np.allclose(post_shifted.se, post_base.se, rtol=1e-10)


def test_epistemic_nearest_neighbors_with_yvar_none():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

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
