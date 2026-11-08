from __future__ import annotations

from dataclasses import dataclass

from enn._rust import validate_multi_objective


@dataclass(frozen=True)
class MultiObjectiveConfig:
    num_metrics: int
    alpha: float = 0.05

    def __post_init__(self) -> None:
        validate_multi_objective(self.num_metrics, self.alpha)
