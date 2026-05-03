from __future__ import annotations

import numpy as np

from enn._rust import hypervolume_2d_max as _rust_hypervolume_2d_max


def hypervolume_2d_max(y: np.ndarray, ref_point: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    if y.size == 0:
        return 0.0
    if y.ndim != 2:
        raise ValueError(y.shape)
    if y.shape[1] != 2:
        raise ValueError(y.shape)
    if ref_point.shape != (2,):
        raise ValueError(ref_point.shape)

    return float(_rust_hypervolume_2d_max(y, ref_point))
