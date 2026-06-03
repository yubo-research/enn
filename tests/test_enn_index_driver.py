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


def test_enn_index_path_requires_hnsw_usearch_driver():
    train_x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    train_y = np.zeros((2, 1), dtype=float)
    with pytest.raises(ValueError, match="index_path requires"):
        EpistemicNearestNeighbors(
            train_x,
            train_y,
            index_driver=ENNIndexDriver.FLAT,
            index_path="/tmp/x.usearch",
        )


def test_enn_index_path_reconciles_stale_smaller_checkpoint(tmp_path):
    """Full in-memory train + smaller on-disk checkpoint must not skip sync."""
    index_path = tmp_path / "index.usearch"
    train2 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    train_y = np.zeros((2, 1), dtype=float)
    try:
        EpistemicNearestNeighbors(
            train2,
            train_y,
            index_driver=ENNIndexDriver.HNSW_USEARCH,
            index_path=str(index_path),
        )
    except ValueError as exc:
        if "usearch" in str(exc).lower():
            pytest.skip("ennbo built without usearch feature")
        raise

    train5 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [5.0, 5.0],
        ],
        dtype=float,
    )
    train_y5 = np.arange(5.0, dtype=float).reshape(5, 1)
    try:
        model = EpistemicNearestNeighbors(
            train5,
            train_y5,
            index_driver=ENNIndexDriver.HNSW_USEARCH,
            index_path=str(index_path),
        )
    except ValueError as exc:
        if "usearch" in str(exc).lower():
            pytest.skip("ennbo built without usearch feature")
        raise

    assert len(model) == 5
    query = np.array([[4.9, 4.9]], dtype=float)
    _, idx = enn_index_neighbor_distances_and_indices(
        model.rust_backend, query, search_k=1, exclude_nearest=False
    )
    assert int(idx[0, 0]) == 4


def test_enn_index_path_rebuilds_when_checkpoint_prefix_differs(tmp_path):
    """Equal-size checkpoint with wrong vectors must not serve stale KNN."""
    index_path = tmp_path / "index.usearch"
    train2 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    train_y = np.zeros((2, 1), dtype=float)
    try:
        EpistemicNearestNeighbors(
            train2,
            train_y,
            index_driver=ENNIndexDriver.HNSW_USEARCH,
            index_path=str(index_path),
        )
    except ValueError as exc:
        if "usearch" in str(exc).lower():
            pytest.skip("ennbo built without usearch feature")
        raise

    train2_new = np.array([[9.0, 9.0], [8.0, 8.0]], dtype=float)
    try:
        model = EpistemicNearestNeighbors(
            train2_new,
            train_y,
            index_driver=ENNIndexDriver.HNSW_USEARCH,
            index_path=str(index_path),
        )
        flat = _enn(train2_new, index_driver=ENNIndexDriver.FLAT)
    except ValueError as exc:
        if "usearch" in str(exc).lower():
            pytest.skip("ennbo built without usearch feature")
        raise

    query = np.array([[9.05, 9.05]], dtype=float)
    _, idx = enn_index_neighbor_distances_and_indices(
        model.rust_backend, query, search_k=1, exclude_nearest=False
    )
    _, flat_idx = enn_index_neighbor_distances_and_indices(
        flat.rust_backend, query, search_k=1, exclude_nearest=False
    )
    assert int(idx[0, 0]) == int(flat_idx[0, 0]) == 0


def test_enn_view_only_reopen_then_equal_count_add_posterior_uses_row_ordinals(
    tmp_path,
):
    """View-only reopen + N adds matching checkpoint size must rebuild, not append."""
    index_path = tmp_path / "index.usearch"
    rng = np.random.default_rng(0)
    n = 50
    train_x = rng.standard_normal((n, 2))
    train_y = np.zeros((n, 1), dtype=float)
    try:
        EpistemicNearestNeighbors(
            train_x,
            train_y,
            index_driver=ENNIndexDriver.HNSW_USEARCH,
            index_path=str(index_path),
        )
        model = EpistemicNearestNeighbors(
            np.empty((0, 2)),
            np.empty((0, 1)),
            index_driver=ENNIndexDriver.HNSW_USEARCH,
            index_path=str(index_path),
        )
    except ValueError as exc:
        if "usearch" in str(exc).lower():
            pytest.skip("ennbo built without usearch feature")
        raise

    add_x = rng.standard_normal((n, 2)) + 100.0
    add_y = np.arange(n, dtype=float).reshape(-1, 1) + 100.0
    model.add(add_x, add_y)
    assert len(model) == n

    query = add_x[:1]
    flat = _enn(add_x, index_driver=ENNIndexDriver.FLAT)
    _, idx = enn_index_neighbor_distances_and_indices(
        model.rust_backend, query, search_k=1, exclude_nearest=False
    )
    _, flat_idx = enn_index_neighbor_distances_and_indices(
        flat.rust_backend, query, search_k=1, exclude_nearest=False
    )
    assert int(idx[0, 0]) == int(flat_idx[0, 0])
    assert int(idx[0, 0]) < n

    from enn.enn.enn_params import ENNParams

    model.posterior(
        query,
        params=ENNParams(
            k_num_neighbors=1,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.0,
        ),
    )


def test_enn_view_only_reopen_then_add_posterior_uses_row_ordinals(tmp_path):
    """Empty train + stale index_path + add must not return USearch keys as row ids."""
    index_path = tmp_path / "index.usearch"
    rng = np.random.default_rng(0)
    n = 100
    train_x = rng.standard_normal((n, 2))
    train_y = np.zeros((n, 1), dtype=float)
    try:
        EpistemicNearestNeighbors(
            train_x,
            train_y,
            index_driver=ENNIndexDriver.HNSW_USEARCH,
            index_path=str(index_path),
        )
        model = EpistemicNearestNeighbors(
            np.empty((0, 2)),
            np.empty((0, 1)),
            index_driver=ENNIndexDriver.HNSW_USEARCH,
            index_path=str(index_path),
        )
    except ValueError as exc:
        if "usearch" in str(exc).lower():
            pytest.skip("ennbo built without usearch feature")
        raise

    assert len(model) == 0
    model.add(np.array([[0.1, 0.2]]), np.array([[1.0]]))
    assert len(model) == 1
    _, idx = enn_index_neighbor_distances_and_indices(
        model.rust_backend,
        np.array([[0.1, 0.2]]),
        search_k=1,
        exclude_nearest=False,
    )
    assert int(idx[0, 0]) == 0

    from enn.enn.enn_params import ENNParams

    model.posterior(
        np.array([[0.1, 0.2]]),
        params=ENNParams(
            k_num_neighbors=1,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.0,
        ),
    )


def test_enn_index_path_file_backed_sync_persists(tmp_path):
    index_path = tmp_path / "index.usearch"
    train_x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    train_y = np.zeros((2, 1), dtype=float)
    try:
        model = EpistemicNearestNeighbors(
            train_x,
            train_y,
            index_driver=ENNIndexDriver.HNSW_USEARCH,
            index_path=str(index_path),
        )
    except ValueError as exc:
        if "usearch" in str(exc).lower():
            pytest.skip("ennbo built without usearch feature")
        raise
    size_after_build = index_path.stat().st_size
    model.add(np.array([[0.0, 1.0]]), np.zeros((1, 1)))
    assert len(model) == 3
    assert index_path.stat().st_size == size_after_build
    model.sync_index()
    assert index_path.stat().st_size > size_after_build
