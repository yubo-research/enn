"""Contract tests for create_optimizer and Optimizer interface."""

from __future__ import annotations

import inspect

import numpy as np

from enn import create_optimizer, turbo_enn_config, turbo_one_config, turbo_zero_config
from enn.turbo.config import ENNSurrogateConfig, lhd_only_config
from enn.turbo.rust_optimizer import RustOptimizer


class TestCreateOptimizerContract:
    """API contract tests for create_optimizer."""

    def test_create_optimizer_signature(self):
        sig = inspect.signature(create_optimizer)
        params = list(sig.parameters.keys())
        assert "bounds" in params
        assert "config" in params
        assert "rng" in params

    def test_returns_optimizer_with_ask_tell_telemetry(self):
        """create_optimizer returns object with ask, tell, telemetry methods."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        config = turbo_one_config(num_init=2)
        rng = np.random.default_rng(42)

        opt = create_optimizer(bounds=bounds, config=config, rng=rng)
        assert hasattr(opt, "ask")
        assert callable(opt.ask)
        assert hasattr(opt, "tell")
        assert callable(opt.tell)
        assert hasattr(opt, "telemetry")
        assert callable(opt.telemetry)

    def test_ask_returns_candidates(self):
        """Optimizer.ask(num_arms) returns array of shape (num_arms, d)."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        config = turbo_one_config(num_init=2)
        rng = np.random.default_rng(42)

        opt = create_optimizer(bounds=bounds, config=config, rng=rng)
        x = opt.ask(num_arms=2)
        assert isinstance(x, np.ndarray)
        assert x.shape == (2, 2)
        assert np.all(x >= bounds[:, 0])
        assert np.all(x <= bounds[:, 1])

    def test_telemetry_returns_telemetry_obj(self):
        """Optimizer.telemetry() returns Telemetry with dt_fit, dt_gen, dt_sel, dt_tell."""
        from enn import Telemetry

        bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        config = turbo_one_config(num_init=2)
        rng = np.random.default_rng(42)

        opt = create_optimizer(bounds=bounds, config=config, rng=rng)
        tele = opt.telemetry()
        assert isinstance(tele, Telemetry)
        assert hasattr(tele, "dt_fit")
        assert hasattr(tele, "dt_gen")
        assert hasattr(tele, "dt_sel")
        assert hasattr(tele, "dt_tell")


class TestCreateOptimizerContractRustBacked:
    """Contract tests for create_optimizer when Rust backend is used (ENN, ZERO, LHD)."""

    def test_default_turbo_enn_uses_rust_backend(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        opt = create_optimizer(
            bounds=bounds, config=turbo_enn_config(), rng=np.random.default_rng(42)
        )
        assert isinstance(opt, RustOptimizer)

    def test_turbo_enn_contract(self):
        """TuRBO-ENN config returns optimizer with correct ask shape and bounds."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        config = turbo_enn_config(enn=ENNSurrogateConfig(k=3), num_init=4)
        rng = np.random.default_rng(99)
        opt = create_optimizer(bounds=bounds, config=config, rng=rng)
        x = opt.ask(num_arms=3)
        assert x.shape == (3, 2)
        assert np.all(x >= bounds[:, 0])
        assert np.all(x <= bounds[:, 1])
        assert hasattr(opt, "telemetry")
        assert callable(opt.telemetry)

    def test_turbo_zero_contract(self):
        """TuRBO-ZERO config returns optimizer with correct ask shape and bounds."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        config = turbo_zero_config(num_init=4)
        rng = np.random.default_rng(101)
        opt = create_optimizer(bounds=bounds, config=config, rng=rng)
        x = opt.ask(num_arms=2)
        assert x.shape == (2, 2)
        assert np.all(x >= bounds[:, 0])
        assert np.all(x <= bounds[:, 1])

    def test_lhd_only_contract(self):
        """LHD_ONLY config returns optimizer with correct ask shape and bounds."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        config = lhd_only_config(num_init=5)
        rng = np.random.default_rng(103)
        opt = create_optimizer(bounds=bounds, config=config, rng=rng)
        x = opt.ask(num_arms=3)
        assert x.shape == (3, 2)
        assert np.all(x >= bounds[:, 0])
        assert np.all(x <= bounds[:, 1])
