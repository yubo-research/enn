from __future__ import annotations

import tempfile

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.turbo.config.enn_index_driver import ENNIndexDriver


def _brute_force_neighbor_ids(
    train_x: np.ndarray, query: np.ndarray, k: int
) -> list[int]:
    dists = np.sum((train_x - query) ** 2, axis=1)
    order = np.argsort(dists, kind="stable")
    return order[:k].tolist()


def test_disk_bpann_batched_sync_recall_above_incore_threshold():
    rng = np.random.default_rng(42)
    n = 15_000
    d = 10
    k = 9
    num_queries = 20
    batch_size = 200

    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 1))

    with tempfile.TemporaryDirectory(prefix="enn_flat_forest_") as work_dir:
        model = EpistemicNearestNeighbors(
            np.empty((0, d)),
            np.empty((0, 1)),
            index_driver=ENNIndexDriver.BPANN_DISK,
            work_dir=work_dir,
            enn_storage="disk",
        )
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            model.add(x[start:end], y[start:end])
        model.ensure_index_sync()

        total_recall = 0.0
        for q in range(num_queries):
            query = x[q]
            got = model.neighbors(query, k=k).tolist()
            expected = _brute_force_neighbor_ids(x, query, k)
            hits = sum(1 for idx in got if idx in expected)
            total_recall += hits / k

        recall = total_recall / num_queries
        assert recall >= 0.90, f"recall@{k} = {recall:.4f}"


def test_disk_bpann_midband_vector_leaf_forest_recall():
    """Mid-band soft-sync (8192 < n <= 10000) uses vector-leaf flat forest."""
    rng = np.random.default_rng(42)
    n = 9_000
    d = 10
    k = 9
    num_queries = 20
    batch_size = 200

    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 1))

    with tempfile.TemporaryDirectory(prefix="enn_midband_") as work_dir:
        model = EpistemicNearestNeighbors(
            np.empty((0, d)),
            np.empty((0, 1)),
            index_driver=ENNIndexDriver.BPANN_DISK,
            work_dir=work_dir,
            enn_storage="disk",
        )
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            model.add(x[start:end], y[start:end])
        model.ensure_index_sync()

        total_recall = 0.0
        for q in range(num_queries):
            query = x[q]
            got = model.neighbors(query, k=k).tolist()
            expected = _brute_force_neighbor_ids(x, query, k)
            hits = sum(1 for idx in got if idx in expected)
            total_recall += hits / k

        recall = total_recall / num_queries
        assert recall >= 0.90, f"mid-band recall@{k} = {recall:.4f}"


def test_disk_bpann_large_scale_batched_recall():
    """Many incremental soft-sync fragments must not be compacted into greedy-only trees."""
    rng = np.random.default_rng(42)
    n = 100_000
    d = 10
    k = 9
    num_queries = 10
    batch_size = 200

    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 1))

    with tempfile.TemporaryDirectory(prefix="enn_large_") as work_dir:
        model = EpistemicNearestNeighbors(
            np.empty((0, d)),
            np.empty((0, 1)),
            index_driver=ENNIndexDriver.BPANN_DISK,
            work_dir=work_dir,
            enn_storage="disk",
        )
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            model.add(x[start:end], y[start:end])
        model.ensure_index_sync()

        total_recall = 0.0
        for q in range(num_queries):
            query = x[q]
            got = model.neighbors(query, k=k).tolist()
            expected = _brute_force_neighbor_ids(x, query, k)
            hits = sum(1 for idx in got if idx in expected)
            total_recall += hits / k

        recall = total_recall / num_queries
        assert recall >= 0.90, f"large-scale recall@{k} = {recall:.4f}"
