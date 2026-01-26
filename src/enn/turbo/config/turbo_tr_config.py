from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..components.protocols import TrustRegion
    from .candidate_rv import CandidateRV


from .tr_length_config import TRLengthConfig


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
