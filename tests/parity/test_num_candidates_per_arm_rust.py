"""Rust-path parity for num_candidates_per_arm pool sizing."""

from __future__ import annotations

import numpy as np
import pytest

from enn import create_optimizer, turbo_enn_config, turbo_zero_config
from enn.turbo.config import CandidateGenConfig, ENNFitConfig, ENNSurrogateConfig
from enn.turbo.rust_optimizer import RustOptimizer
from enn.turbo.rust_optimizer_helpers import _config_to_rust_overrides

try:
    from enn import _rust  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def test_num_candidates_per_arm_rust_optimizer_ask_tell():
    config = turbo_zero_config(num_candidates_per_arm=1500, num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(99)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert isinstance(opt, RustOptimizer)
    assert config.candidates.resolve_num_candidates(num_dim=2, num_arms=3) == 4500
    while opt.init_progress is not None:
        x = opt.ask(num_arms=3)
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        opt.tell(x, y)
    x = opt.ask(num_arms=3)
    assert opt.telemetry().num_candidates == 4500
    y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
    opt.tell(x, y)
    x2 = opt.ask(num_arms=8)
    assert x2.shape == (8, 2)
    assert opt.telemetry().num_candidates == 12000


def test_rust_overrides_pass_num_candidates_per_arm():
    cfg_enn = turbo_enn_config(
        candidates=CandidateGenConfig(num_candidates_per_arm=25),
        num_init=4,
    )
    cfg_zero = turbo_zero_config(num_candidates_per_arm=25, num_init=4)
    for cfg in (cfg_enn, cfg_zero):
        overrides = _config_to_rust_overrides(cfg)
        assert overrides is not None
        assert overrides.get("num_candidates_per_arm") == 25


def test_enn_per_arm_only_runtime_matches_python_resolve():
    config = turbo_enn_config(
        candidates=CandidateGenConfig(num_candidates_per_arm=40),
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=8)),
        num_init=4,
    )
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(101)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert isinstance(opt, RustOptimizer)
    expected = config.candidates.resolve_num_candidates(num_dim=2, num_arms=3)
    assert expected == 200
    while opt.init_progress is not None:
        x = opt.ask(num_arms=3)
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        opt.tell(x, y)
    opt.ask(num_arms=3)
    assert opt.telemetry().num_candidates == expected


def test_fixed_and_per_arm_rust_runtime_matches_python_resolve():
    config = turbo_zero_config(
        num_candidates=100,
        num_candidates_per_arm=50,
        num_init=4,
    )
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(102)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert config.candidates.resolve_num_candidates(num_dim=2, num_arms=4) == 200
    while opt.init_progress is not None:
        x = opt.ask(num_arms=4)
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        opt.tell(x, y)
    opt.ask(num_arms=4)
    assert opt.telemetry().num_candidates == 200


def test_python_and_rust_pool_formula_agree_for_per_arm():
    cfg = turbo_zero_config(num_candidates_per_arm=40, num_init=4)
    assert cfg.candidates.resolve_num_candidates(num_dim=2, num_arms=3) == 200
    assert cfg.candidates.resolve_num_candidates(num_dim=2, num_arms=8) == 320
    overrides = _config_to_rust_overrides(cfg)
    assert overrides.get("num_candidates_per_arm") == 40
    assert overrides.get("num_candidates_factor") == 100.0
    assert overrides.get("max_candidates") == 5000


def _finish_init(opt: RustOptimizer, num_arms: int) -> None:
    while opt.init_progress is not None:
        x = opt.ask(num_arms=num_arms)
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        opt.tell(x, y)


def test_default_num_candidates_runtime_matches_python_resolve_many_arms():
    """Rust num_arms*10 floor must not exceed Python default resolve."""
    num_arms = 25
    config = turbo_zero_config(num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = create_optimizer(bounds=bounds, config=config, rng=np.random.default_rng(103))
    assert isinstance(opt, RustOptimizer)
    expected = config.candidates.resolve_num_candidates(num_dim=2, num_arms=num_arms)
    assert expected == 200
    _finish_init(opt, num_arms)
    opt.ask(num_arms=num_arms)
    assert opt.telemetry().num_candidates == expected


def test_per_arm_only_high_dim_runtime_matches_python_resolve():
    """Per-arm-only overrides must cap base like Python min(5000, 100*dim)."""
    num_dim = 60
    num_arms = 1
    config = turbo_zero_config(num_candidates_per_arm=40, num_init=4)
    bounds = np.tile(np.array([[0.0, 1.0]], dtype=float), (num_dim, 1))
    opt = create_optimizer(bounds=bounds, config=config, rng=np.random.default_rng(104))
    assert isinstance(opt, RustOptimizer)
    expected = config.candidates.resolve_num_candidates(
        num_dim=num_dim, num_arms=num_arms
    )
    assert expected == 5000
    _finish_init(opt, num_arms)
    opt.ask(num_arms=num_arms)
    assert opt.telemetry().num_candidates == expected
