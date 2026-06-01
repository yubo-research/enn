from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_class_support import enn_index_neighbor_distances_and_indices
from enn.turbo.config.enn_index_driver import ENNIndexDriver, ENN_INDEX_DRIVER_TO_RUST


def _enn(train_x, *, index_driver=ENNIndexDriver.FLAT):
    train_y = np.zeros((train_x.shape[0], 1), dtype=float)
    return EpistemicNearestNeighbors(
        train_x, train_y, scale_x=False, index_driver=index_driver
    )


def test_enn_index_driver_to_rust_maps_all_three():
    assert set(ENNIndexDriver) == set(ENN_INDEX_DRIVER_TO_RUST.keys())
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.FLAT] == "exact"
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.HNSW] == "hnsw"
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.HNSW_USEARCH] == "hnsw_usearch"


def test_enn_hnsw_usearch_driver_without_feature_raises():
    """hnsw_usearch is accepted at the API boundary but fails if usearch is not linked."""
    train_x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    train_y = np.zeros((2, 1), dtype=float)
    try:
        EpistemicNearestNeighbors(
            train_x,
            train_y,
            index_driver=ENNIndexDriver.HNSW_USEARCH,
        )
    except ValueError as exc:
        assert "usearch" in str(exc).lower()
    else:
        pytest.skip("ennbo built with usearch feature")


def test_enn_index_driver_flat_hnsw_metamorphic_neighbor_set():
    """On a tiny fixture, Faiss Flat and Faiss HNSW should return identical neighbor indices."""
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    query = np.array([[0.25, 0.25]], dtype=float)
    search_k = 2
    flat = _enn(train_x, index_driver=ENNIndexDriver.FLAT)
    hnsw = _enn(train_x, index_driver=ENNIndexDriver.HNSW)
    _, flat_idx = enn_index_neighbor_distances_and_indices(
        flat.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    _, hnsw_idx = enn_index_neighbor_distances_and_indices(
        hnsw.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    np.testing.assert_array_equal(flat_idx, hnsw_idx)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_enn_index_driver_neighbor_indices_fuzz(seed: int):
    rng = np.random.default_rng(seed)
    n_train, dim, search_k = 20, 4, 3
    train_x = rng.uniform(0.0, 1.0, size=(n_train, dim))
    query = rng.uniform(0.0, 1.0, size=(1, dim))
    for driver in (ENNIndexDriver.FLAT, ENNIndexDriver.HNSW):
        enn = _enn(train_x, index_driver=driver)
        _, idx = enn_index_neighbor_distances_and_indices(
            enn.rust_backend, query, search_k=search_k, exclude_nearest=False
        )
        assert idx.shape == (1, search_k)
        assert np.all(idx >= 0)
        assert np.all(idx < n_train)
    print(f"index_driver_neighbor_indices_fuzz seed={seed}")
