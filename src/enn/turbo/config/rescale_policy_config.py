from __future__ import annotations

from dataclasses import dataclass
from .rescalarize import Rescalarize


@dataclass(frozen=True)
class RescalePolicyConfig:
    rescalarize: Rescalarize = Rescalarize.ON_PROPOSE
