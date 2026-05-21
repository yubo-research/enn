from __future__ import annotations

from typing import Any


def validate_optimizer_config(cfg: Any) -> None:
    from .acquisition import (
        DrawAcquisitionConfig,
        NDSOptimizerConfig,
        ParetoAcquisitionConfig,
        UCBAcquisitionConfig,
    )
    from .surrogate import NoSurrogateConfig

    if type(cfg.init.init_strategy).__name__ == "LHDOnlyInit":
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
