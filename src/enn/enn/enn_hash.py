from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from enn._rust import normal_hash_batch_multi_seed as _rust_hash
from enn._rust import normal_hash_batch_multi_seed_fast as _rust_hash_fast


def normal_hash_batch_multi_seed(
    function_seeds: np.ndarray, data_indices: np.ndarray, num_metrics: int
) -> np.ndarray:
    import numpy as np

    function_seeds = np.asarray(function_seeds, dtype=np.int64)
    data_indices = np.asarray(data_indices)
    if num_metrics <= 0:
        raise ValueError(num_metrics)
    return np.asarray(_rust_hash(function_seeds, data_indices, num_metrics), dtype=float)


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
