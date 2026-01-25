from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .rescalarize import Rescalarize
from .turbo_tr_config import TRLengthConfig

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..components.protocols import TrustRegion
    from .enums import CandidateRV

from .enums import CandidateRV


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


@dataclass(frozen=True)
class RescalePolicyConfig:
    rescalarize: Rescalarize = Rescalarize.ON_PROPOSE


@dataclass(frozen=True)
class MorboTRConfig:
    multi_objective: MultiObjectiveConfig
    length: TRLengthConfig = TRLengthConfig()
    rescale_policy: RescalePolicyConfig = RescalePolicyConfig()
    noise_aware: bool = False

    @property
    def rescalarize(self) -> Rescalarize:
        return self.rescale_policy.rescalarize

    @property
    def num_metrics(self) -> int:
        return self.multi_objective.num_metrics

    @property
    def alpha(self) -> float:
        return self.multi_objective.alpha

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
        candidate_rv: CandidateRV = CandidateRV.SOBOL,
    ) -> TrustRegion:
        from ..morbo_trust_region import MorboTrustRegion

        return MorboTrustRegion(
            config=self,
            num_dim=num_dim,
            rng=rng,
            candidate_rv=candidate_rv,
        )
