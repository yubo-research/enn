from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .turbo_config import TurboConfig
    from .turbo_mode import TurboMode
    from .turbo_mode_impl import TurboModeImpl


def get_mode_registry() -> dict:
    from .turbo_config import (
        LHDOnlyConfig,
        TurboENNConfig,
        TurboOneConfig,
        TurboZeroConfig,
    )
    from .turbo_mode import TurboMode

    return {
        TurboMode.TURBO_ONE: (TurboOneConfig, "turbo_one_impl", "TurboOneImpl"),
        TurboMode.TURBO_ZERO: (TurboZeroConfig, "turbo_zero_impl", "TurboZeroImpl"),
        TurboMode.TURBO_ENN: (TurboENNConfig, "turbo_enn_impl", "TurboENNImpl"),
        TurboMode.LHD_ONLY: (LHDOnlyConfig, "lhd_only_impl", "LHDOnlyImpl"),
    }


def make_impl(mode: TurboMode, config: TurboConfig) -> TurboModeImpl:
    import importlib

    registry = get_mode_registry()
    if mode not in registry:
        raise ValueError(f"Unknown mode: {mode}")

    config_class, module_name, class_name = registry[mode]
    if not isinstance(config, config_class):
        raise ValueError(
            f"mode={mode} requires {config_class.__name__}, got {type(config).__name__}"
        )

    module = importlib.import_module(f".{module_name}", package=__package__)
    impl_class = getattr(module, class_name)
    return impl_class(config)


def get_default_config(mode: TurboMode) -> TurboConfig:
    registry = get_mode_registry()
    if mode not in registry:
        raise ValueError(f"Unknown mode: {mode}")
    config_class, _, _ = registry[mode]
    return config_class()


def validate_config(mode: TurboMode, config: TurboConfig | None) -> TurboConfig:
    if config is None:
        return get_default_config(mode)
    registry = get_mode_registry()
    if mode not in registry:
        raise ValueError(f"Unknown mode: {mode}")
    config_class, _, _ = registry[mode]
    if not isinstance(config, config_class):
        raise ValueError(
            f"mode={mode} requires {config_class.__name__}, got {type(config).__name__}"
        )
    return config
