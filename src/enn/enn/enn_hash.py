from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from enn._rust import normal_hash_batch_multi_seed_fast as _rust_hash_fast


def normal_hash_batch_multi_seed(
    function_seeds: np.ndarray, data_indices: np.ndarray, num_metrics: int
) -> np.ndarray:
    import numpy as np
    from scipy.special import ndtri

    num_seeds = len(function_seeds)
    unique_indices, inverse = np.unique(data_indices, return_inverse=True)
    num_unique = len(unique_indices)
    seed_grid, idx_grid, metric_grid = np.meshgrid(
        function_seeds.astype(np.uint64),
        unique_indices.astype(np.uint64),
        np.arange(num_metrics, dtype=np.uint64),
        indexing="ij",
    )
    seed_flat = seed_grid.ravel()
    idx_flat = idx_grid.ravel()
    metric_flat = metric_grid.ravel()
    combined_seeds = (seed_flat * np.uint64(1_000_003) + idx_flat) * np.uint64(
        1_000_003
    ) + metric_flat
    uniform_vals = np.empty(len(combined_seeds), dtype=float)
    for i, seed in enumerate(combined_seeds):
        rng = np.random.Generator(np.random.Philox(int(seed)))
        uniform_vals[i] = rng.random()
    uniform_vals = np.clip(uniform_vals, 1e-10, 1.0 - 1e-10)
    normal_vals = ndtri(uniform_vals).reshape(num_seeds, num_unique, num_metrics)
    return normal_vals[:, inverse.ravel(), :].reshape(
        num_seeds, *data_indices.shape, num_metrics
    )


def normal_hash_batch_multi_seed_fast(
    function_seeds: np.ndarray, data_indices: np.ndarray, num_metrics: int
) -> np.ndarray:
    import numpy as np

    function_seeds = np.asarray(function_seeds, dtype=np.int64)
    data_indices = np.asarray(data_indices)
    if num_metrics <= 0:
        raise ValueError(num_metrics)
    return np.asarray(
        _rust_hash_fast(function_seeds, data_indices, num_metrics), dtype=float
    )
