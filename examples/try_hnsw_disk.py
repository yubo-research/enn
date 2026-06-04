"""Minimal smoke for the in-tree disk HNSW backend (`hnsw_disk`).

Run from the repository root after `maturin develop`::

    PYTHONPATH=src python -m examples.try_hnsw_disk
"""

from __future__ import annotations

import tempfile

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from enn.turbo.config.enn_index_driver import ENNIndexDriver


def main() -> None:
    rng = np.random.default_rng(42)
    dim, n = 8, 200
    train_x = rng.standard_normal((n, dim))
    train_y = rng.standard_normal((n, 1))
    query = rng.standard_normal((1, dim))

    with tempfile.TemporaryDirectory(prefix="enn_hnsw_disk_") as work_dir:
        model = EpistemicNearestNeighbors(
            train_x[:100],
            train_y[:100],
            index_driver=ENNIndexDriver.HNSW_DISK,
            enn_storage="disk",
            work_dir=work_dir,
        )
        for i in range(100, n):
            model.add(train_x[i : i + 1], train_y[i : i + 1])
        model.ensure_index_sync()
        out = model.posterior(
            query,
            params=ENNParams(
                k_num_neighbors=5,
                epistemic_variance_scale=1.0,
                aleatoric_variance_scale=0.0,
            ),
        )
        assert out.mu.shape == (1, 1)
        print(f"work_dir={work_dir} n={len(model)} posterior ok")


if __name__ == "__main__":
    main()
