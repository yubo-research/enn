from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .morbo_tr_config import MorboTRConfig, MultiObjectiveConfig, RescalePolicyConfig
from .no_tr_config import NoTRConfig
from .turbo_tr_config import TRLengthConfig, TurboTRConfig

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..components.protocols import TrustRegion
    from .enums import CandidateRV


class TrustRegionConfig(Protocol):
    def build(
        self,
        *,
        num_dim: int,
        rng: Generator,
        candidate_rv: CandidateRV,
    ) -> TrustRegion: ...


__all__ = [
    "MorboTRConfig",
    "MultiObjectiveConfig",
    "NoTRConfig",
    "RescalePolicyConfig",
    "TRLengthConfig",
    "TrustRegionConfig",
    "TurboTRConfig",
]
