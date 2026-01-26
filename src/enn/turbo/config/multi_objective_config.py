from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class MultiObjectiveConfig:
    num_metrics: int
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.num_metrics < 2:
            raise ValueError(
                f"num_metrics must be >= 2 for MORBO, got {self.num_metrics}"
            )
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
