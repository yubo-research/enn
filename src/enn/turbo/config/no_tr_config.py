from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..components.protocols import TrustRegion
    from .candidate_rv import CandidateRV


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
        from ..components.builder import build_trust_region

        return build_trust_region(self, num_dim, rng, candidate_rv)
