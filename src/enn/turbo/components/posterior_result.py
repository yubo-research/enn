from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class PosteriorResult:
    mu: np.ndarray
    sigma: np.ndarray | None = None
