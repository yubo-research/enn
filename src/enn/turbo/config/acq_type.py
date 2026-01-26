from __future__ import annotations
from enum import Enum


class AcqType(Enum):
    THOMPSON = "thompson"
    PARETO = "pareto"
    UCB = "ucb"
