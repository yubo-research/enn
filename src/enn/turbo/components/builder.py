from __future__ import annotations
from typing import TYPE_CHECKING
from .acquisition import HnRAcqOptimizer, ThompsonAcqOptimizer, UCBAcqOptimizer

if TYPE_CHECKING:
    from ..config.optimizer_config import OptimizerConfig
    from .protocols import AcquisitionOptimizer, Surrogate


def build_surrogate(config: OptimizerConfig) -> Surrogate:
    return config.surrogate.build()


def build_acquisition_optimizer(config: OptimizerConfig) -> AcquisitionOptimizer:
    from ..config.acquisition import HnROptimizerConfig

    base = config.acquisition.build()
    if isinstance(config.acq_optimizer, HnROptimizerConfig):
        if isinstance(base, (ThompsonAcqOptimizer, UCBAcqOptimizer)):
            return HnRAcqOptimizer(base)
        raise ValueError(f"HnR not supported with {type(base).__name__}")
    return base
