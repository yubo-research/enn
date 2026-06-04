from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_class_support import enn_index_neighbor_distances_and_indices
from enn.turbo.config.enn_index_driver import ENNIndexDriver, ENN_INDEX_DRIVER_TO_RUST


def _enn(train_x, *, index_driver=ENNIndexDriver.FLAT, work_dir=None):
    train_y = np.zeros((train_x.shape[0], 1), dtype=float)
    kwargs: dict = {"scale_x": False, "index_driver": index_driver}
    if work_dir is not None:
        kwargs["work_dir"] = work_dir
        kwargs["enn_storage"] = "disk"
    return EpistemicNearestNeighbors(train_x, train_y, **kwargs)


def test_enn_index_driver_to_rust_maps_all():
    assert set(ENNIndexDriver) == set(ENN_INDEX_DRIVER_TO_RUST.keys())
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.FLAT] == "exact"
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.HNSW] == "hnsw"
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.HNSW_DISK] == "hnsw_disk"


def test_enn_hnsw_disk_in_memory_raises():
    train_x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    train_y = np.zeros((2, 1), dtype=float)
    with pytest.raises(ValueError, match="work_dir|ENN_WORK_DIR"):
        EpistemicNearestNeighbors(
            train_x,
            train_y,
            index_driver=ENNIndexDriver.HNSW_DISK,
            enn_storage="disk",
        )


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


def test_enn_work_dir_requires_disk_driver():
    train_x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    train_y = np.zeros((2, 1), dtype=float)
    with pytest.raises(ValueError, match="Disk storage requires IndexDriver::HNSWDisk"):
        EpistemicNearestNeighbors(
            train_x,
            train_y,
            index_driver=ENNIndexDriver.FLAT,
            work_dir="/tmp/enn_work",
            enn_storage="disk",
        )


def test_enn_disk_backend_persists_observation_files(tmp_path):
    work_dir = tmp_path / "persist"
    train_x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    np.zeros((2, 1), dtype=float)
    model = _enn(
        train_x,
        index_driver=ENNIndexDriver.HNSW_DISK,
        work_dir=str(work_dir),
    )
    assert (work_dir / "train_x.bin").exists()
    size_before = (work_dir / "train_x.bin").stat().st_size
    model.add(np.array([[0.0, 1.0]]), np.zeros((1, 1)))
    assert len(model) == 3
    assert (work_dir / "train_x.bin").stat().st_size > size_before


@pytest.mark.parametrize("driver", [ENNIndexDriver.HNSW_DISK])
def test_enn_disk_backend_incremental_add_and_search_hnsw_disk(tmp_path, driver):
    work_dir = tmp_path / "enn_disk_hnsw"
    rng = np.random.default_rng(42)
    n, d, init = 60, 4, 50
    train_x = rng.standard_normal((n, d))
    train_y = rng.standard_normal((n, 1))
    model = EpistemicNearestNeighbors(
        train_x[:init],
        train_y[:init],
        index_driver=driver,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    model.add(train_x[init:], train_y[init:])
    assert len(model) == n
    model.ensure_index_sync()

    flat = _enn(train_x, index_driver=ENNIndexDriver.FLAT)
    for qi in range(10):
        query = train_x[qi : qi + 1]
        _, idx = enn_index_neighbor_distances_and_indices(
            model.rust_backend, query, search_k=1, exclude_nearest=False
        )
        _, flat_idx = enn_index_neighbor_distances_and_indices(
            flat.rust_backend, query, search_k=1, exclude_nearest=False
        )
        assert int(idx[0, 0]) == int(flat_idx[0, 0]), f"query row {qi}"


@pytest.mark.parametrize("driver", [ENNIndexDriver.HNSW_DISK])
def test_enn_disk_backend_train_rows_at_matches_memory_hnsw_disk(tmp_path, driver):
    work_dir = tmp_path / "enn_disk_rows_hnsw"
    rng = np.random.default_rng(7)
    n, d = 30, 3
    train_x = rng.standard_normal((n, d))
    mem = _enn(train_x, index_driver=ENNIndexDriver.FLAT)
    disk = _enn(train_x, index_driver=driver, work_dir=str(work_dir))
    indices = rng.choice(n, size=10, replace=False).tolist()
    mx, my, _ = mem.train_rows_at(indices)
    dx, dy, _ = disk.train_rows_at(indices)
    np.testing.assert_allclose(mx, dx)
    np.testing.assert_allclose(my, dy)
