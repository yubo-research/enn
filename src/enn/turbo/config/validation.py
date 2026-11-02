from __future__ import annotations

from typing import Any


def validate_optimizer_config(cfg: Any) -> None:
    from enn._rust import validate_optimizer_rules

    from .acquisition import acquisition_kind
    from .init_strategies import LHDOnlyInit
    from .surrogate import NoSurrogateConfig
    from .trust_region import NoTRConfig

    validate_optimizer_rules(
        {
            "lhd_only": isinstance(cfg.init.init_strategy, LHDOnlyInit),
            "has_surrogate": not isinstance(cfg.surrogate, NoSurrogateConfig),
            "has_trust_region": not isinstance(cfg.trust_region, NoTRConfig),
        },
        acquisition_kind(cfg.acquisition),
    )
