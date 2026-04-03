from __future__ import annotations

from dataclasses import dataclass

from .tr_length_config import TRLengthConfig


@dataclass(frozen=True)
class TurboTRConfig:
    length: TRLengthConfig = TRLengthConfig()
    noise_aware: bool = False

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max
