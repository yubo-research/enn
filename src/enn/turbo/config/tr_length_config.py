from __future__ import annotations

from dataclasses import dataclass

from enn._rust import validate_length_overrides


@dataclass(frozen=True)
class TRLengthConfig:
    length_init: float | None = None
    length_min: float | None = None
    length_max: float | None = None

    def __post_init__(self) -> None:
        validate_length_overrides(self.length_init, self.length_min, self.length_max)
