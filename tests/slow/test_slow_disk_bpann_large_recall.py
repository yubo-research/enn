from __future__ import annotations

import tempfile

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors


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
            expected = np.argsort(np.sum((x - query) ** 2, axis=1), kind="stable")[
                :k
            ].tolist()
            hits = sum(1 for idx in got if idx in expected)
            total_recall += hits / k

        recall = total_recall / num_queries
        assert recall >= 0.85, f"large-scale recall@{k} = {recall:.4f}"
