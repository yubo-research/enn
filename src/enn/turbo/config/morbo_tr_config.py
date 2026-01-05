from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .rescalarize import Rescalarize

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..components.protocols import TrustRegion


@dataclass(frozen=True)
class MorboTRConfig:
    num_metrics: int
    alpha: float = 0.05
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    rescalarize: Rescalarize = Rescalarize.ON_PROPOSE

    def __post_init__(self) -> None:
        if self.num_metrics < 2:
            raise ValueError(
                f"num_metrics must be >= 2 for MORBO, got {self.num_metrics}"
            )
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.length_init <= 0:
            raise ValueError(f"length_init must be > 0, got {self.length_init}")
        if self.length_min <= 0:
            raise ValueError(f"length_min must be > 0, got {self.length_min}")
        if self.length_max <= 0:
            raise ValueError(f"length_max must be > 0, got {self.length_max}")

    def build(
        self,
        *,
        num_dim: int,
        rng: Generator,
    ) -> TrustRegion:
        from ..morbo_trust_region import MorboTrustRegion

        return MorboTrustRegion(
            config=self,
            num_dim=num_dim,
            rng=rng,
        )
