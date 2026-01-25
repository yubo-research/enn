from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..components.protocols import TrustRegion
    from .enums import CandidateRV


@dataclass(frozen=True)
class NoTRConfig:
    noise_aware: bool = False

    def build(
        self,
        *,
        num_dim: int,
        rng: Generator,
        candidate_rv: CandidateRV | None = None,
    ) -> TrustRegion:
        from ..components.incumbent_selector import ScalarIncumbentSelector
        from ..no_trust_region import NoTrustRegion

        return NoTrustRegion(
            config=self,
            num_dim=num_dim,
            incumbent_selector=ScalarIncumbentSelector(noise_aware=self.noise_aware),
        )
