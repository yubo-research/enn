from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass
class SurrogateResult:
    model: Any
    lengthscales: np.ndarray | None = None
