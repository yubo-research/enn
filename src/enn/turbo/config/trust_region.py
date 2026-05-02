from __future__ import annotations

from typing import Union

from .init_strategy_base import InitStrategy
from .morbo_tr_config import MorboTRConfig, MultiObjectiveConfig, RescalePolicyConfig
from .no_tr_config import NoTRConfig
from .turbo_tr_config import TRLengthConfig, TurboTRConfig

TrustRegionConfig = Union[NoTRConfig, TurboTRConfig, MorboTRConfig]


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
