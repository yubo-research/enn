from __future__ import annotations

from enn.turbo.turbo_config import (
    LHDOnlyConfig,
    TurboConfig,
    TurboENNConfig,
    TurboOneConfig,
    TurboZeroConfig,
)
from enn.turbo.build_components import build_impl
from enn.turbo.lhd_only_impl import LHDOnlyImpl
from enn.turbo.turbo_enn_impl import TurboENNImpl
from enn.turbo.turbo_one_impl import TurboOneImpl
from enn.turbo.turbo_zero_impl import TurboZeroImpl


def test_build_impl_turbo_one():
    config = TurboOneConfig()
    impl = build_impl(config)
    assert isinstance(impl, TurboOneImpl)


def test_build_impl_turbo_zero():
    config = TurboZeroConfig()
    impl = build_impl(config)
    assert isinstance(impl, TurboZeroImpl)


def test_build_impl_turbo_enn():
    config = TurboENNConfig()
    impl = build_impl(config)
    assert isinstance(impl, TurboENNImpl)


def test_build_impl_lhd_only():
    config = LHDOnlyConfig()
    impl = build_impl(config)
    assert isinstance(impl, LHDOnlyImpl)


def test_build_impl_base_config_uses_no_surrogate():
    config = TurboConfig()
    impl = build_impl(config)
    assert isinstance(impl, TurboZeroImpl)
