from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass
class PrepareAskResult:
    model: Any
    y_mean: float | None
    y_std: float | None
    lengthscales: np.ndarray | None
