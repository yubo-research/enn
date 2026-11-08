from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from evals.harness.work_dirs import fresh_work_dir


def test_neighbors_returns_correct_number_and_ordering():
    import conftest

    model, train_x, train_y, _, _ = conftest.make_enn_model()
    indices = model.neighbors(np.zeros((1, 3), dtype=float), k=5, exclude_nearest=False)
    assert indices.shape == (1, 5)
    assert np.all((0 <= indices) & (indices < len(train_x)))
    distances = [np.linalg.norm(train_x[i]) for i in indices[0]]
    assert all(
        distances[i] <= distances[i + 1] + 1e-6 for i in range(len(distances) - 1)
    )
    assert train_y[indices[0]].shape == (5, 1)


def test_neighbors_store_returns_valid_indices():
    rng = np.random.default_rng(3)
    n, d = 35, 4
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True).astype(float)
    model = EpistemicNearestNeighbors(
        train_x,
        train_y,
        work_dir=fresh_work_dir(),
    )
    x_query = rng.standard_normal((1, d))
    neighbors = model.neighbors(x_query, k=6, exclude_nearest=False)
    assert neighbors.shape == (1, 6)
    assert np.all((0 <= neighbors) & (neighbors < n))


def test_neighbors_exclude_nearest():
    import conftest

    model, train_x, _train_y, _train_yvar, _rng = conftest.make_enn_model()
    x_query = train_x[5:6].copy()
    neighbors_exclude = model.neighbors(x_query, k=5, exclude_nearest=True)
    neighbors_include = model.neighbors(x_query, k=5, exclude_nearest=False)
    assert neighbors_exclude.shape == (1, 5) and neighbors_include.shape == (1, 5)
    assert neighbors_include[0, 0] == 5
    assert neighbors_exclude[0, 0] != 5
    assert 5 not in set(neighbors_exclude[0].tolist())


def test_neighbors_with_empty_observations():
    d = 3
    train_x = np.zeros((0, d), dtype=float)
    train_y = np.zeros((0, 1), dtype=float)
    train_yvar = np.ones((0, 1), dtype=float)
    model = EpistemicNearestNeighbors(
        train_x, train_y, train_yvar, work_dir=fresh_work_dir()
    )
    x_query = np.zeros((1, d), dtype=float)
    neighbors = model.neighbors(x_query, k=5, exclude_nearest=False)
    assert neighbors.shape == (1, 0)


def test_neighbors_k_larger_than_available():
    rng = np.random.default_rng(0)
    n, d = 5, 3
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True).astype(float)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(
        train_x, train_y, train_yvar, work_dir=fresh_work_dir()
    )
    x_query = np.zeros((1, d), dtype=float)
    neighbors = model.neighbors(x_query, k=20, exclude_nearest=False)
    assert neighbors.shape == (1, n)


def test_neighbors_k_zero():
    import conftest

    model, _train_x, _train_y, _train_yvar, _rng = conftest.make_enn_model(n=10)
    x_query = np.zeros((1, 3), dtype=float)
    neighbors = model.neighbors(x_query, k=0, exclude_nearest=False)
    assert neighbors.shape == (1, 0)


def test_neighbors_with_multiple_metrics():
    rng = np.random.default_rng(0)
    n, d = 15, 3
    train_x = rng.standard_normal((n, d))
    train_y = rng.standard_normal((n, 2))
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(
        train_x, train_y, train_yvar, work_dir=fresh_work_dir()
    )
    x_query = np.zeros((1, d), dtype=float)
    neighbors = model.neighbors(x_query, k=5, exclude_nearest=False)
    assert neighbors.shape == (1, 5)
    assert np.all((0 <= neighbors) & (neighbors < n))


def test_neighbors_handles_a_batch_of_queries():
    import conftest

    model, _train_x, _train_y, _train_yvar, _rng = conftest.make_enn_model(n=10)
    queries = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    batch = model.neighbors(queries, k=3, exclude_nearest=False)
    assert batch.shape == (2, 3)
    for i in range(2):
        single = model.neighbors(queries[i : i + 1], k=3, exclude_nearest=False)
        np.testing.assert_array_equal(batch[i : i + 1], single)


def test_neighbors_rejects_one_dimensional_query():
    import conftest

    model, _train_x, _train_y, _train_yvar, _rng = conftest.make_enn_model(n=10)
    with pytest.raises(ValueError, match=r"x must be 2-dimensional, got shape \[3\]"):
        model.neighbors(np.zeros(3, dtype=float), k=3)


def test_neighbors_exclude_nearest_requires_multiple_observations():
    rng = np.random.default_rng(0)
    d = 3
    train_x = rng.standard_normal((1, d))
    train_y, train_yvar = np.array([[1.0]]), np.array([[0.1]])
    model = EpistemicNearestNeighbors(
        train_x, train_y, train_yvar, work_dir=fresh_work_dir()
    )
    x_query = np.zeros((1, d), dtype=float)
    with pytest.raises(
        ValueError, match="exclude_nearest=true requires at least 2 observations"
    ):
        model.neighbors(x_query, k=1, exclude_nearest=True)
