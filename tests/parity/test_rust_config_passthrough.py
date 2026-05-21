"""Regression tests for Python→Rust config pass-through (plan Phase 1)."""

from __future__ import annotations

import pytest

from enn.turbo.config import ENNFitConfig, ENNSurrogateConfig, turbo_enn_config
from enn.turbo.rust_optimizer_helpers import _config_to_rust_overrides

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def test_scale_x_true_passed_in_rust_overrides():
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(
            k=3,
            scale_x=True,
            fit=ENNFitConfig(num_fit_samples=10),
        ),
        num_init=4,
    )
    overrides = _config_to_rust_overrides(config)
    assert overrides is not None
    assert overrides.get("scale_x") is True
