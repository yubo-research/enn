from __future__ import annotations

import numpy as np

from enn._rust import separable_unimodal as _separable


def separable_unimodal_objective(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    return np.asarray(_separable(x), dtype=float)
