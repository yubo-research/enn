from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class GPFitResult:
    model: Any
    likelihood: Any
    y_mean: Any
    y_std: Any
