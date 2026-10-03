from __future__ import annotations

import numpy as np


def _to_rust_seeds(function_seeds: np.ndarray | list[int]) -> list[int]:
    if hasattr(function_seeds, "__iter__"):
        return np.asarray(function_seeds, dtype=np.int64).tolist()
    return list(function_seeds)


def enn_index_neighbor_distances_and_indices(
    rust_model,
    x: np.ndarray,
    *,
    search_k: int,
    exclude_nearest: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dist2s, idx = rust_model.index_neighbor_distances_and_indices(
        np.asarray(x, dtype=float),
        int(search_k),
        bool(exclude_nearest),
    )
    return np.asarray(dist2s, dtype=float), np.asarray(idx, dtype=int)


def enn_neighbor_distances_and_indices(
    rust_model,
    x: np.ndarray,
    *,
    search_k: int,
    exclude_nearest: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dist2s, idx = rust_model.neighbor_distances_and_indices(
        np.asarray(x, dtype=float),
        int(search_k),
        bool(exclude_nearest),
    )
    return np.asarray(dist2s, dtype=float), np.asarray(idx, dtype=int)
