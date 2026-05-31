from __future__ import annotations

import numpy as np


def separable_unimodal_objective(x: np.ndarray) -> np.ndarray:
    x1 = x[:, 0]
    x2 = x[:, 1]
    y1 = 500_000.0 - 8.0 * (x1 - 120.0) ** 2
    y2 = 12.5 - 110.0 * (x2 - 0.91) ** 2
    return np.stack([y1, y2], axis=1)
