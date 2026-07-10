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
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.BPANN_DISK] == "bpann_disk"


def test_enn_bpann_disk_in_memory_raises():
    train_x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    train_y = np.zeros((2, 1), dtype=float)
    with pytest.raises(ValueError, match="work_dir|ENN_WORK_DIR"):
        EpistemicNearestNeighbors(
            train_x,
            train_y,
            index_driver=ENNIndexDriver.BPANN_DISK,
            enn_storage="disk",
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_enn_index_driver_neighbor_indices_fuzz(seed: int):
    rng = np.random.default_rng(seed)
    n_train, dim, search_k = 20, 4, 3
    train_x = rng.uniform(0.0, 1.0, size=(n_train, dim))
    query = rng.uniform(0.0, 1.0, size=(1, dim))
    enn = _enn(train_x, index_driver=ENNIndexDriver.FLAT)
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
    with pytest.raises(
        ValueError, match="Disk storage requires IndexDriver::BpAnnDisk"
    ):
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
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
    )
    assert (work_dir / "train_x.bin").exists()
    model.add(np.array([[0.0, 1.0]]), np.zeros((1, 1)))
    assert len(model) == 3
    x_at, _, _ = model.train_rows_at([2])
    assert x_at.shape == (1, 2)
    np.testing.assert_allclose(x_at[0], [0.0, 1.0])


def test_enn_disk_bpann_posterior_with_pending_matches_fresh(tmp_path):
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(42)
    d = 3
    work_dir = tmp_path / "enn_disk_pending_post"
    x0 = rng.standard_normal((10, d))
    y0 = rng.standard_normal((10, 1))
    x1 = rng.standard_normal((5, d))
    y1 = rng.standard_normal((5, 1))

    inc = EpistemicNearestNeighbors(
        x0,
        y0,
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    inc.ensure_index_sync()
    inc.add(x1, y1)

    fresh = EpistemicNearestNeighbors(
        np.vstack([x0, x1]),
        np.vstack([y0, y1]),
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(tmp_path / "enn_disk_fresh"),
        enn_storage="disk",
    )
    fresh.ensure_index_sync()
    params = ENNParams(
        k_num_neighbors=3, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    x_test = rng.standard_normal((4, d))
    post_inc = inc.posterior(x_test, params=params)
    post_fresh = fresh.posterior(x_test, params=params)
    np.testing.assert_allclose(post_inc.mu, post_fresh.mu, rtol=1e-5)
    np.testing.assert_allclose(post_inc.se, post_fresh.se, rtol=1e-5)


def test_enn_disk_bpann_reopen_uses_persisted_num_metrics(tmp_path):
    """Reopen must not trust placeholder empty train_y column count for num_outputs."""
    from enn.enn.enn_params import ENNParams

    work_dir = tmp_path / "enn_disk_reopen_num_metrics"
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    train_y = np.array([[0.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=float)
    params = ENNParams(
        k_num_neighbors=2, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    query = np.array([[0.5, 0.5]], dtype=float)

    EpistemicNearestNeighbors(
        train_x,
        train_y,
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )

    reopened = EpistemicNearestNeighbors(
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 1), dtype=float),
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    assert reopened.num_outputs == 2
    post = reopened.posterior(query, params=params)
    assert post.mu.shape == (1, 2)
    np.testing.assert_allclose(post.mu[0], [0.5, 1.5], rtol=1e-5)


def test_enn_disk_bpann_empty_store_reopen_uses_persisted_num_metrics(tmp_path):
    """Empty disk store reopen must sync num_outputs from metadata, not placeholder train_y."""
    work_dir = tmp_path / "enn_disk_empty_reopen_num_metrics"
    EpistemicNearestNeighbors(
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 2), dtype=float),
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    reopened = EpistemicNearestNeighbors(
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 1), dtype=float),
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    assert reopened.num_outputs == 2
    assert len(reopened) == 0


def test_enn_disk_bpann_reopen_scale_x_posterior_matches_fresh_without_sync(tmp_path):
    """Disk reopen with scale_x=true must not return wrong posterior before ensure_index_sync."""
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(7)
    d = 3
    work_dir = tmp_path / "enn_disk_scale_x_reopen"
    x = rng.standard_normal((15, d))
    y = rng.standard_normal((15, 1))

    model = EpistemicNearestNeighbors(
        x,
        y,
        scale_x=True,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    model.ensure_index_sync()
    del model

    reopened = EpistemicNearestNeighbors(
        np.zeros((0, d)),
        np.zeros((0, 1)),
        scale_x=True,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    fresh = EpistemicNearestNeighbors(
        x,
        y,
        scale_x=True,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(tmp_path / "enn_disk_scale_x_reopen_fresh"),
        enn_storage="disk",
    )
    fresh.ensure_index_sync()
    params = ENNParams(
        k_num_neighbors=3, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    x_test = rng.standard_normal((4, d))
    post_reopen = reopened.posterior(x_test, params=params)
    post_fresh = fresh.posterior(x_test, params=params)
    np.testing.assert_allclose(post_reopen.mu, post_fresh.mu, rtol=1e-5)
    np.testing.assert_allclose(post_reopen.se, post_fresh.se, rtol=1e-5)


def test_enn_disk_bpann_posterior_scale_x_pending_matches_fresh(tmp_path):
    """Phase D: scale_x pending leg uses live scale; posterior matches fresh."""
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(7)
    d = 3
    work_dir = tmp_path / "enn_disk_scale_x_pending"
    x = rng.standard_normal((15, d))
    y = rng.standard_normal((15, 1))

    inc = EpistemicNearestNeighbors(
        x,
        y,
        scale_x=True,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )

    fresh = EpistemicNearestNeighbors(
        x,
        y,
        scale_x=True,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(tmp_path / "enn_disk_scale_x_fresh"),
        enn_storage="disk",
    )
    fresh.ensure_index_sync()
    params = ENNParams(
        k_num_neighbors=3, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    x_test = rng.standard_normal((4, d))
    post_inc = inc.posterior(x_test, params=params)
    post_fresh = fresh.posterior(x_test, params=params)
    np.testing.assert_allclose(post_inc.mu, post_fresh.mu, rtol=1e-5)
    np.testing.assert_allclose(post_inc.se, post_fresh.se, rtol=1e-5)


def _build_multi_batch_disk_persist_model(work_dir, d, rng):
    x_batches = [
        rng.standard_normal((1000, d)),
        rng.standard_normal((1000, d)),
        rng.standard_normal((500, d)),
    ]
    y_batches = [rng.standard_normal((n, 1)) for n in (1000, 1000, 500)]
    model = EpistemicNearestNeighbors(
        np.empty((0, d)),
        np.empty((0, 1)),
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    for x_chunk, y_chunk in zip(x_batches, y_batches, strict=True):
        model.add(x_chunk, y_chunk)
        model.schedule_background_flush()
    model.ensure_index_sync()
    return model, np.vstack(x_batches), np.vstack(y_batches)


def test_enn_disk_persist_index_multi_batch_reopen_matches_posterior(tmp_path):
    """Multi-batch disk store: persist preserves in-session posterior; reopen matches reference."""
    import json

    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(17)
    d = 32
    work_dir = tmp_path / "enn_disk_persist"
    model, x_all, y_all = _build_multi_batch_disk_persist_model(work_dir, d, rng)
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    x_test = rng.standard_normal((4, d))
    pre_persist = model.posterior(x_test, params=params)
    model.persist_index_to_disk()
    post_persist = model.posterior(x_test, params=params)
    np.testing.assert_allclose(pre_persist.mu, post_persist.mu, rtol=1e-5)
    np.testing.assert_allclose(pre_persist.se, post_persist.se, rtol=1e-5)

    header = json.loads((work_dir / "index" / "header.json").read_text())
    assert header["indexed_rows"] == 2500

    fresh = EpistemicNearestNeighbors(
        x_all,
        y_all,
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(tmp_path / "enn_disk_persist_fresh"),
        enn_storage="disk",
    )
    fresh.ensure_index_sync()
    post_fresh = fresh.posterior(x_test, params=params)

    reopened = EpistemicNearestNeighbors(
        np.empty((0, d)),
        np.empty((0, 1)),
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    reopened.ensure_index_sync()
    post_reopen = reopened.posterior(x_test, params=params)
    np.testing.assert_allclose(post_reopen.mu, post_fresh.mu, rtol=1e-5)
    np.testing.assert_allclose(post_reopen.se, post_fresh.se, rtol=1e-5)


@pytest.mark.parametrize("driver", [ENNIndexDriver.BPANN_DISK])
def test_enn_disk_backend_incremental_add_and_search_bpann_disk(tmp_path, driver):
    work_dir = tmp_path / f"enn_disk_{driver.name.lower()}"
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


@pytest.mark.parametrize("driver", [ENNIndexDriver.BPANN_DISK])
def test_enn_disk_backend_train_rows_at_matches_memory_bpann_disk(tmp_path, driver):
    work_dir = tmp_path / f"enn_disk_rows_{driver.name.lower()}"
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
