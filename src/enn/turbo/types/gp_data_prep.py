from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class GPDataPrep:
    train_x: Any
    train_y: Any
    is_multi: bool
    y_mean: Any
    y_std: Any
    y_raw: Any
