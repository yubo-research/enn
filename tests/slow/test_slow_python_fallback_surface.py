from __future__ import annotations

import importlib

import pytest

from enn.turbo.python_fallback import gp_surface


@pytest.mark.slow
def test_production_modules_importable():
    for mod in gp_surface.PRODUCTION_MODULES:
        importlib.import_module(f"enn.turbo.python_fallback.{mod}")
