from __future__ import annotations

import pytest

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def test_optimizer_quality_ci_subset():
    from .parity_quality_gate import assert_ci_quality_gate

    assert_ci_quality_gate()
