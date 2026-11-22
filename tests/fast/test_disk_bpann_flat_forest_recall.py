from __future__ import annotations

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors


def _brute_force_neighbor_ids(
    train_x: np.ndarray, query: np.ndarray, k: int
) -> list[int]:
    dists = np.sum((train_x - query) ** 2, axis=1)
    order = np.argsort(dists, kind="stable")
    return order[:k].tolist()


def test_disk_bpann_batched_sync_recall_above_incore_threshold(tmp_path):
    rng = np.random.default_rng(42)
    n = 15_000
    d = 10
    k = 9
    num_queries = 20
    batch_size = 200

    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 1))

    work_dir = str(tmp_path)
    model = EpistemicNearestNeighbors(
        np.empty((0, d)),
        np.empty((0, 1)),
        work_dir=work_dir,
    )
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        model.add(x[start:end], y[start:end])
    model.ensure_index_sync()

    total_recall = 0.0
    for q in range(num_queries):
        query = x[q]
        got = model.neighbors(query[np.newaxis, :], k=k)[0].tolist()
        expected = _brute_force_neighbor_ids(x, query, k)
        hits = sum(1 for idx in got if idx in expected)
        total_recall += hits / k

    recall = total_recall / num_queries
    assert recall >= 0.90, f"recall@{k} = {recall:.4f}"


def test_disk_bpann_midband_vector_leaf_forest_recall(tmp_path):
    """Mid-band soft-sync (8192 < n <= 10000) uses vector-leaf flat forest."""
    rng = np.random.default_rng(42)
    n = 9_000
    d = 10
    k = 9
    num_queries = 20
    batch_size = 200

    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 1))

    work_dir = str(tmp_path)
    model = EpistemicNearestNeighbors(
        np.empty((0, d)),
        np.empty((0, 1)),
        work_dir=work_dir,
    )
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        model.add(x[start:end], y[start:end])
    model.ensure_index_sync()

    total_recall = 0.0
    for q in range(num_queries):
        query = x[q]
        got = model.neighbors(query[np.newaxis, :], k=k)[0].tolist()
        expected = _brute_force_neighbor_ids(x, query, k)
        hits = sum(1 for idx in got if idx in expected)
        total_recall += hits / k

    recall = total_recall / num_queries
    assert recall >= 0.90, f"mid-band recall@{k} = {recall:.4f}"


def test_disk_bpann_large_scale_batched_recall(tmp_path):
    """Many incremental soft-sync fragments must not be compacted into greedy-only trees."""
    rng = np.random.default_rng(42)
    n = 100_000
    d = 10
    k = 9
    num_queries = 10
    batch_size = 200

    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 1))

    work_dir = str(tmp_path)
    model = EpistemicNearestNeighbors(
        np.empty((0, d)),
        np.empty((0, 1)),
        work_dir=work_dir,
    )
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        model.add(x[start:end], y[start:end])
    model.ensure_index_sync()

    total_recall = 0.0
    for q in range(num_queries):
        query = x[q]
        got = model.neighbors(query[np.newaxis, :], k=k)[0].tolist()
        expected = _brute_force_neighbor_ids(x, query, k)
        hits = sum(1 for idx in got if idx in expected)
        total_recall += hits / k

    recall = total_recall / num_queries
    assert recall >= 0.90, f"large-scale recall@{k} = {recall:.4f}"
