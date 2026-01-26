from __future__ import annotations
from enum import Enum, auto


class AcqType(Enum):
    THOMPSON = "thompson"
    PARETO = "pareto"
    UCB = "ucb"


class ENNIndexDriver(Enum):
    FLAT = auto()
    HNSW = auto()


class Rescalarize(Enum):
    ON_RESTART = "on_restart"
    ON_PROPOSE = "on_propose"
