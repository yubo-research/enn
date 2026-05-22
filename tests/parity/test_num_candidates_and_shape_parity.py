"""Parity tests for num_candidates semantics and x/y shape validation."""

from __future__ import annotations

import numpy as np
import pytest

from enn import create_optimizer, turbo_zero_config
from enn.turbo.rust_optimizer import RustOptimizer, is_rust_supported_config

try:
    from enn import _rust  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def test_const_num_candidates_uses_rust_when_constant():
    """Fixed num_candidates maps to Rust min=max override."""
    config = turbo_zero_config(num_candidates=500, num_init=4)
    assert is_rust_supported_config(config)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert isinstance(opt, RustOptimizer)


def test_default_num_candidates_uses_rust_when_available():
    """Default num_candidates (min(5000, 100*dim)) maps to Rust; uses Rust backend."""
    config = turbo_zero_config(num_init=4)  # num_candidates=None -> default
    assert is_rust_supported_config(config)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert isinstance(opt, RustOptimizer)


def test_default_num_candidates_telemetry_matches_python_resolve():
    num_arms = 8
    config = turbo_zero_config(num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = create_optimizer(bounds=bounds, config=config, rng=np.random.default_rng(44))
    assert isinstance(opt, RustOptimizer)
    expected = config.candidates.resolve_num_candidates(num_dim=2, num_arms=num_arms)
    while opt.init_progress is not None:
        x = opt.ask(num_arms=num_arms)
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        opt.tell(x, y)
    opt.ask(num_arms=num_arms)
    assert opt.telemetry().num_candidates == expected


def test_rust_optimizer_tell_raises_on_mismatched_xy_rows():
    """RustOptimizer.tell raises ValueError when x and y have mismatched row counts."""
    config = turbo_zero_config(num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert isinstance(opt, RustOptimizer)

    x = opt.ask(num_arms=3)
    y = np.array([[1.0], [2.0]])  # 2 rows, x has 3
    with pytest.raises(ValueError, match="shape"):
        opt.tell(x, y)
