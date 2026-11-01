from __future__ import annotations

import numpy as np

from enn._rust import hypervolume_2d_max as _rust_hypervolume_2d_max


def hypervolume_2d_max(y: np.ndarray, ref_point: np.ndarray) -> float:
    return float(
        _rust_hypervolume_2d_max(
            np.asarray(y, dtype=float), np.asarray(ref_point, dtype=float)
        )
    )
