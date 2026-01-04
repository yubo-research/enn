from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UCBAcquisitionConfig:
    pass


@dataclass(frozen=True)
class DrawAcquisitionConfig:
    pass


@dataclass(frozen=True)
class ParetoAcquisitionConfig:
    pass


@dataclass(frozen=True)
class RandomAcquisitionConfig:
    pass


AcquisitionConfig = (
    UCBAcquisitionConfig
    | DrawAcquisitionConfig
    | ParetoAcquisitionConfig
    | RandomAcquisitionConfig
)


@dataclass(frozen=True)
class RAASPOptimizerConfig:
    pass


@dataclass(frozen=True)
class NDSOptimizerConfig:
    pass


AcqOptimizerConfig = RAASPOptimizerConfig | NDSOptimizerConfig
