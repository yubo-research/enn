from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .morbo_tr_config import MorboTRConfig, MultiObjectiveConfig, RescalePolicyConfig
from .no_tr_config import NoTRConfig
from .turbo_tr_config import TRLengthConfig, TurboTRConfig

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..components.protocols import TrustRegion


class TrustRegionConfig(Protocol):
    """Protocol for trust region configuration classes."""

    def build(
        self,
        *,
        num_dim: int,
        rng: Generator,
    ) -> TrustRegion:
        """Build the runtime TrustRegion instance."""
        ...


__all__ = [
    "MorboTRConfig",
    "MultiObjectiveConfig",
    "NoTRConfig",
    "RescalePolicyConfig",
    "TRLengthConfig",
    "TrustRegionConfig",
    "TurboTRConfig",
]
