from __future__ import annotations
from typing import Any


def validate_optimizer_config(cfg: Any) -> None:
    from .acquisition import (
        DrawAcquisitionConfig,
        HnROptimizerConfig,
        NDSOptimizerConfig,
        ParetoAcquisitionConfig,
        UCBAcquisitionConfig,
    )
    from .init_strategies import LHDOnlyInit
    from .surrogate import GPSurrogateConfig, NoSurrogateConfig

    if isinstance(cfg.init.init_strategy, LHDOnlyInit):
        if not isinstance(cfg.surrogate, NoSurrogateConfig):
            raise ValueError(
                "init_strategy='lhd_only' requires NoSurrogateConfig surrogate"
            )
    if isinstance(cfg.surrogate, NoSurrogateConfig):
        if isinstance(cfg.acquisition, DrawAcquisitionConfig):
            raise ValueError(
                "DrawAcquisitionConfig (Thompson sampling) requires a surrogate. "
                "NoSurrogateConfig is not compatible with DrawAcquisitionConfig."
            )
        if isinstance(cfg.acquisition, UCBAcquisitionConfig):
            raise ValueError(
                "UCBAcquisitionConfig requires a surrogate. "
                "NoSurrogateConfig is not compatible with UCBAcquisitionConfig."
            )
    if isinstance(cfg.acquisition, ParetoAcquisitionConfig):
        if not isinstance(cfg.acq_optimizer, NDSOptimizerConfig):
            raise ValueError("ParetoAcquisitionConfig requires NDSOptimizerConfig")
    if isinstance(cfg.acq_optimizer, HnROptimizerConfig):
        if isinstance(cfg.acquisition, ParetoAcquisitionConfig):
            raise ValueError(
                "HnROptimizerConfig is not compatible with ParetoAcquisitionConfig"
            )
        if isinstance(cfg.surrogate, GPSurrogateConfig) and isinstance(
            cfg.acquisition, DrawAcquisitionConfig
        ):
            raise NotImplementedError(
                "GP surrogate with DrawAcquisitionConfig and HnROptimizerConfig is not yet implemented"
            )
