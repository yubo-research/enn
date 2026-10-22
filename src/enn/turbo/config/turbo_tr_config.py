from __future__ import annotations

from dataclasses import dataclass, field

from .tr_length_config import TRLengthConfig


@dataclass(frozen=True)
class TurboTRConfig:
    length: TRLengthConfig = field(default_factory=TRLengthConfig)
    noise_aware: bool = False

    @property
    def length_init(self) -> float | None:
        return self.length.length_init

    @property
    def length_min(self) -> float | None:
        return self.length.length_min

    @property
    def length_max(self) -> float | None:
        return self.length.length_max
