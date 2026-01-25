from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PosteriorFlags:
    exclude_nearest: bool = False
    observation_noise: bool = False
