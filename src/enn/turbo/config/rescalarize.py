from __future__ import annotations
from enum import Enum


class Rescalarize(Enum):
    ON_RESTART = "on_restart"
    ON_PROPOSE = "on_propose"
