from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .turbo_config import TurboConfig
    from .turbo_mode_impl import TurboModeImpl


def build_impl(config: TurboConfig) -> TurboModeImpl:
    from .lhd_only_impl import LHDOnlyImpl
    from .turbo_config import (
        ENNSurrogateConfig,
        GPSurrogateConfig,
        NoSurrogateConfig,
    )
    from .turbo_enn_impl import TurboENNImpl
    from .turbo_one_impl import TurboOneImpl
    from .turbo_zero_impl import TurboZeroImpl

    if config.init.init_strategy == "lhd_only":
        return LHDOnlyImpl(config)

    if isinstance(config.surrogate, GPSurrogateConfig):
        return TurboOneImpl(config)

    if isinstance(config.surrogate, ENNSurrogateConfig):
        return TurboENNImpl(config)

    if isinstance(config.surrogate, NoSurrogateConfig):
        return TurboZeroImpl(config)

    raise ValueError(
        f"Unknown surrogate config type: {type(config.surrogate).__name__}"
    )
