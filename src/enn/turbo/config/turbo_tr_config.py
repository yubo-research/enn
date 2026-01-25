from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..components.protocols import TrustRegion
    from .enums import CandidateRV


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

    def build(
        self,
        *,
        num_dim: int,
        rng: Generator,
        candidate_rv: CandidateRV | None = None,
    ) -> TrustRegion:
        from ..components.incumbent_selector import ScalarIncumbentSelector
        from ..turbo_trust_region import TurboTrustRegion

        return TurboTrustRegion(
            config=self,
            num_dim=num_dim,
            incumbent_selector=ScalarIncumbentSelector(noise_aware=self.noise_aware),
        )
