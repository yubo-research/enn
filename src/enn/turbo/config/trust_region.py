from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from .morbo_tr_config import MorboTRConfig, MultiObjectiveConfig, RescalePolicyConfig
from .no_tr_config import NoTRConfig
from .turbo_tr_config import TRLengthConfig, TurboTRConfig

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from ..components.protocols import TrustRegion
    from .candidate_rv import CandidateRV
    from ..strategies import OptimizationStrategy


class TrustRegionConfig(Protocol):
    def build(
        self,
        *,
        num_dim: int,
        rng: Generator,
        candidate_rv: CandidateRV,
    ) -> TrustRegion: ...


class InitStrategy(ABC):
    @abstractmethod
    def create_runtime_strategy(
        self,
        *,
        bounds: np.ndarray,
        rng: Generator,
        num_init: int | None,
    ) -> OptimizationStrategy: ...


__all__ = [
    "InitStrategy",
    "MorboTRConfig",
    "MultiObjectiveConfig",
    "NoTRConfig",
    "RescalePolicyConfig",
    "TRLengthConfig",
    "TrustRegionConfig",
    "TurboTRConfig",
]
