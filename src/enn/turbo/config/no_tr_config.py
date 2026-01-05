from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..components.protocols import TrustRegion


@dataclass(frozen=True)
class NoTRConfig:
    def build(
        self,
        *,
        num_dim: int,
        rng: Generator,  # noqa: ARG002
    ) -> TrustRegion:
        from ..no_trust_region import NoTrustRegion

        return NoTrustRegion(config=self, num_dim=num_dim)
