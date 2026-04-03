from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoTRConfig:
    noise_aware: bool = False
