from __future__ import annotations

from enum import Enum, auto


class TurboMode(Enum):
    TURBO_ONE = auto()
    TURBO_ZERO = auto()
    TURBO_ENN = auto()
    LHD_ONLY = auto()
