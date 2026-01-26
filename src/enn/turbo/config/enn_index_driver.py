from __future__ import annotations
from enum import Enum, auto


class ENNIndexDriver(Enum):
    FLAT = auto()
    HNSW = auto()
