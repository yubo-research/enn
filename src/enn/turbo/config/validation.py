from __future__ import annotations

from typing import Any


def validate_optimizer_config(cfg: Any) -> None:
    from enn._rust import validate_optimizer_rules

    from .acquisition import (
        DrawAcquisitionConfig,
        NDSOptimizerConfig,
        ParetoAcquisitionConfig,
        UCBAcquisitionConfig,
    )
    from .surrogate import NoSurrogateConfig

    if isinstance(cfg.acquisition, UCBAcquisitionConfig):
        acquisition = "ucb"
    elif isinstance(cfg.acquisition, DrawAcquisitionConfig):
        acquisition = "thompson"
    elif isinstance(cfg.acquisition, ParetoAcquisitionConfig):
        acquisition = "pareto"
    else:
        acquisition = "random"
    validate_optimizer_rules(
        [
            type(cfg.init.init_strategy).__name__ == "LHDOnlyInit",
            not isinstance(cfg.surrogate, NoSurrogateConfig),
            isinstance(cfg.acq_optimizer, NDSOptimizerConfig),
        ],
        acquisition,
    )
