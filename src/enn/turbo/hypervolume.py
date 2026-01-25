from __future__ import annotations
import numpy as np


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
    mask = (y[:, 0] > ref_point[0]) & (y[:, 1] > ref_point[1])
    y = y[mask]
    if y.size == 0:
        return 0.0
    order = np.argsort(y[:, 0], kind="mergesort")[::-1]
    y = y[order]
    hv = 0.0
    best_y1 = ref_point[1]
    for i in range(len(y)):
        x0, y1 = y[i]
        if y1 > best_y1:
            best_y1 = y1
        x_next = y[i + 1, 0] if i + 1 < len(y) else ref_point[0]
        hv += (x0 - x_next) * (best_y1 - ref_point[1])
    return float(hv)
