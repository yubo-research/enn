from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .rescalarize import Rescalarize
from .turbo_tr_config import TRLengthConfig
from .candidate_rv import CandidateRV
from .multi_objective_config import MultiObjectiveConfig
from .rescale_policy_config import RescalePolicyConfig

if TYPE_CHECKING:
    from numpy.random import Generator
    from ..components.protocols import TrustRegion


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
        from ..components.builder import build_trust_region

        return build_trust_region(self, num_dim, rng, candidate_rv)
