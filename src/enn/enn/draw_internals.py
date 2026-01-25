from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class DrawInternals:
    idx: np.ndarray
    w_normalized: np.ndarray
    l2: np.ndarray
    mu: np.ndarray
    se: np.ndarray
