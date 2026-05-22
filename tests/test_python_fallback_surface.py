from __future__ import annotations

import importlib

from enn.turbo.python_fallback import gp_surface
from enn.turbo.python_fallback.components.builder import build_surrogate
from enn.turbo.config import turbo_enn_config, turbo_one_config, turbo_zero_config


def test_build_surrogate_accepts_gp_only():
    assert isinstance(build_surrogate(turbo_one_config()), object)


def test_build_surrogate_rejects_rust_owned_configs():
    import pytest

    for cfg in (turbo_enn_config(), turbo_zero_config()):
        with pytest.raises(ValueError, match="Rust optimizer"):
            build_surrogate(cfg)


def test_production_modules_importable():
    for mod in gp_surface.PRODUCTION_MODULES:
        importlib.import_module(f"enn.turbo.python_fallback.{mod}")
