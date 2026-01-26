from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationHistoryConfig:
    trailing_obs: int | None = None

    def __post_init__(self) -> None:
        if self.trailing_obs is not None and self.trailing_obs <= 0:
            raise ValueError(f"trailing_obs must be > 0, got {self.trailing_obs}")
