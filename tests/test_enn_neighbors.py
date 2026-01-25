from __future__ import annotations
import numpy as np
import pytest
from enn.enn.enn_class import EpistemicNearestNeighbors


def test_neighbors_returns_correct_number_and_ordering():
    import conftest

    model, train_x, train_y, _, _ = conftest.make_enn_model()
    neighbors = model.neighbors(np.zeros(3, dtype=float), k=5, exclude_nearest=False)
    assert len(neighbors) == 5
    indices = []
    for x_n, y_n in neighbors:
        assert x_n.shape == (3,) and y_n.shape == (1,)
        matches = np.all(np.abs(train_x - x_n[np.newaxis, :]) < 1e-6, axis=1)
        assert np.any(matches), "Neighbor x not found in training data"
        idx = np.where(matches)[0][0]
        indices.append(idx)
        assert np.allclose(train_y[idx], y_n), "Neighbor y doesn't match training data"
    distances = [np.linalg.norm(train_x[i]) for i in indices]
    assert all(
        distances[i] <= distances[i + 1] + 1e-6 for i in range(len(distances) - 1)
    )


def test_neighbors_exclude_nearest():
    import conftest

    model, train_x, _train_y, _train_yvar, _rng = conftest.make_enn_model()
    x_query = train_x[5].copy()
    neighbors_exclude = model.neighbors(x_query, k=5, exclude_nearest=True)
    neighbors_include = model.neighbors(x_query, k=5, exclude_nearest=False)
    assert len(neighbors_exclude) == 5 and len(neighbors_include) == 5
    assert np.allclose(neighbors_include[0][0], x_query)
    assert not np.allclose(neighbors_exclude[0][0], x_query)


def test_neighbors_with_empty_observations():
    d = 3
    train_x = np.zeros((0, d), dtype=float)
    train_y = np.zeros((0, 1), dtype=float)
    train_yvar = np.ones((0, 1), dtype=float)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    x_query = np.zeros(d, dtype=float)
    neighbors = model.neighbors(x_query, k=5, exclude_nearest=False)
    assert neighbors == []


def test_neighbors_k_larger_than_available():
    rng = np.random.default_rng(0)
    n, d = 5, 3
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True).astype(float)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    x_query = np.zeros(d, dtype=float)
    neighbors = model.neighbors(x_query, k=20, exclude_nearest=False)
    assert len(neighbors) == n


def test_neighbors_k_zero():
    import conftest

    model, _train_x, _train_y, _train_yvar, _rng = conftest.make_enn_model(n=10)
    x_query = np.zeros(3, dtype=float)
    neighbors = model.neighbors(x_query, k=0, exclude_nearest=False)
    assert neighbors == []


def test_neighbors_with_multiple_metrics():
    rng = np.random.default_rng(0)
    n, d = 15, 3
    train_x = rng.standard_normal((n, d))
    train_y = rng.standard_normal((n, 2))
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    x_query = np.zeros(d, dtype=float)
    neighbors = model.neighbors(x_query, k=5, exclude_nearest=False)
    assert len(neighbors) == 5
    for x_neighbor, y_neighbor in neighbors:
        assert x_neighbor.shape == (d,) and y_neighbor.shape == (2,)


def test_neighbors_accepts_2d_input():
    import conftest

    model, _train_x, _train_y, _train_yvar, _rng = conftest.make_enn_model(n=10)
    d = 3
    x_query_1d = np.zeros(d, dtype=float)
    neighbors_1d = model.neighbors(x_query_1d, k=3, exclude_nearest=False)
    x_query_2d = np.zeros((1, d), dtype=float)
    neighbors_2d = model.neighbors(x_query_2d, k=3, exclude_nearest=False)
    assert len(neighbors_1d) == len(neighbors_2d) == 3
    for (x1, y1), (x2, y2) in zip(neighbors_1d, neighbors_2d):
        assert np.allclose(x1, x2) and np.allclose(y1, y2)


def test_neighbors_exclude_nearest_requires_multiple_observations():
    rng = np.random.default_rng(0)
    d = 3
    train_x = rng.standard_normal((1, d))
    train_y, train_yvar = np.array([[1.0]]), np.array([[0.1]])
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    x_query = np.zeros(d, dtype=float)
    with pytest.raises(
        ValueError, match="exclude_nearest=True requires at least 2 observations"
    ):
        model.neighbors(x_query, k=1, exclude_nearest=True)
