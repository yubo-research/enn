from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TRLengthConfig:
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6

    def __post_init__(self) -> None:
        if self.length_init <= 0:
            raise ValueError(f"length_init must be > 0, got {self.length_init}")
        if self.length_min <= 0:
            raise ValueError(f"length_min must be > 0, got {self.length_min}")
        if self.length_max <= 0:
            raise ValueError(f"length_max must be > 0, got {self.length_max}")
        if self.length_min >= self.length_max:
            raise ValueError(
                f"length_min must be < length_max, got {self.length_min} >= {self.length_max}"
            )
        if self.length_init > self.length_max:
            raise ValueError(
                f"length_init must be <= length_max, got {self.length_init} > {self.length_max}"
            )
        if self.length_min > self.length_init:
            raise ValueError(
                f"length_min must be <= length_init, got {self.length_min} > {self.length_init}"
            )
