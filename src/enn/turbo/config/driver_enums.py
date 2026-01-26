from __future__ import annotations
from enum import Enum, auto


class RAASPDriver(Enum):
    ORIG = auto()
    FAST = auto()


class CandidateRV(Enum):
    SOBOL = "sobol"
    UNIFORM = "uniform"
