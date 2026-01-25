from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


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
    num_seeds = len(function_seeds)
    unique_indices, inverse = np.unique(data_indices, return_inverse=True)

    def _splitmix64(x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            x = x + np.uint64(0x9E3779B97F4A7C15)
            z = x
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
            return z

    seeds_u64 = function_seeds.astype(np.uint64, copy=False)
    unique_u64 = unique_indices.astype(np.uint64, copy=False)
    metric_u64 = np.arange(num_metrics, dtype=np.uint64)
    normal_vals = np.empty((num_seeds, unique_indices.size, num_metrics), dtype=float)
    p = np.uint64(1_000_003)
    inv_2p53 = 1.0 / 9007199254740992.0
    for si, s in enumerate(seeds_u64):
        with np.errstate(over="ignore"):
            base = (s * p + unique_u64) * p
        combined = base[:, None] + metric_u64[None, :]
        r1 = _splitmix64(combined)
        r2 = _splitmix64(combined ^ np.uint64(0xD2B74407B1CE6E93))
        u1 = (r1 >> np.uint64(11)).astype(np.float64) * inv_2p53
        u2 = (r2 >> np.uint64(11)).astype(np.float64) * inv_2p53
        u1 = np.clip(u1, 1e-12, 1.0 - 1e-12)
        normal_vals[si, :, :] = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return normal_vals[:, inverse.ravel(), :].reshape(
        num_seeds, *data_indices.shape, num_metrics
    )
