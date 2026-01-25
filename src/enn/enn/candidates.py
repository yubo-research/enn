from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Candidates:
    dist2: np.ndarray
    ids: np.ndarray
    y: np.ndarray
    yvar: np.ndarray | None
