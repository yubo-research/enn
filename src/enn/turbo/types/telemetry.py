from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Telemetry:
    dt_fit: float
    dt_sel: float
    dt_gen: float = 0.0
    dt_tell: float = 0.0
