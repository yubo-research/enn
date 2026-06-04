"""Brute-force k-NN helper for hannoy smoke.

Run the full smoke script::

    python -m examples.hannoy_smoke_runner
"""

from __future__ import annotations

import numpy as np


def brute_knn(query: np.ndarray, data: np.ndarray, k: int) -> list[tuple[int, float]]:
    dists = np.sum((data - query) ** 2, axis=1)
    idx = np.argsort(dists)[:k]
    return [(int(i), float(dists[i])) for i in idx]
