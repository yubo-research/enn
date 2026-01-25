from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class TellInputs:
    x: np.ndarray
    y: np.ndarray
    y_var: np.ndarray | None
    num_metrics: int
