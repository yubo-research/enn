from __future__ import annotations

import pytest

from enn.turbo.turbo_mode import TurboMode
from enn.turbo.turbo_config import (
    TurboOneConfig,
    TurboZeroConfig,
    TurboENNConfig,
    LHDOnlyConfig,
)
from enn.turbo.turbo_mode_registry import (
    get_mode_registry,
    make_impl,
    get_default_config,
    validate_config,
)


def test_get_mode_registry_returns_all_modes():
    registry = get_mode_registry()
    assert TurboMode.TURBO_ONE in registry
    assert TurboMode.TURBO_ZERO in registry
    assert TurboMode.TURBO_ENN in registry
    assert TurboMode.LHD_ONLY in registry


def test_make_impl_turbo_one():
    config = TurboOneConfig()
    impl = make_impl(TurboMode.TURBO_ONE, config)
    assert impl is not None


def test_make_impl_turbo_zero():
    config = TurboZeroConfig()
    impl = make_impl(TurboMode.TURBO_ZERO, config)
    assert impl is not None


def test_make_impl_turbo_enn():
    config = TurboENNConfig()
    impl = make_impl(TurboMode.TURBO_ENN, config)
    assert impl is not None


def test_make_impl_lhd_only():
    config = LHDOnlyConfig()
    impl = make_impl(TurboMode.LHD_ONLY, config)
    assert impl is not None


def test_make_impl_wrong_config_raises():
    with pytest.raises(ValueError, match="requires"):
        make_impl(TurboMode.TURBO_ONE, TurboENNConfig())


def test_get_default_config_returns_correct_types():
    assert isinstance(get_default_config(TurboMode.TURBO_ONE), TurboOneConfig)
    assert isinstance(get_default_config(TurboMode.TURBO_ZERO), TurboZeroConfig)
    assert isinstance(get_default_config(TurboMode.TURBO_ENN), TurboENNConfig)
    assert isinstance(get_default_config(TurboMode.LHD_ONLY), LHDOnlyConfig)


def test_validate_config_returns_default_when_none():
    result = validate_config(TurboMode.TURBO_ONE, None)
    assert isinstance(result, TurboOneConfig)


def test_validate_config_returns_config_when_valid():
    config = TurboOneConfig(num_init=10)
    result = validate_config(TurboMode.TURBO_ONE, config)
    assert result is config


def test_validate_config_raises_on_wrong_type():
    with pytest.raises(ValueError, match="requires"):
        validate_config(TurboMode.TURBO_ONE, TurboENNConfig())
