from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class NeighborData:
    dist2s: np.ndarray
    idx: np.ndarray
    y_neighbors: np.ndarray
    k: int
