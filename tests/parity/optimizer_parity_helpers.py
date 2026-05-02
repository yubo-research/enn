"""Shared helpers for optimizer Rust vs Python parity tests."""

from __future__ import annotations

import numpy as np

from enn import create_optimizer
from enn.turbo.rust_optimizer import is_rust_supported_config


def assert_rust_optimizer_tr_obs_after_cycles(
    bounds: np.ndarray,
    config,
    *,
    opt_seed: int,
    cycle_rng_seed: int,
    obj_fn,
) -> None:
    rng = np.random.default_rng(cycle_rng_seed)
    opt = get_rust_optimizer(bounds, config, seed=opt_seed)
    x0 = opt.ask(num_arms=2)
    assert opt.tr_obs_count == 0
    y0 = obj_fn(x0).reshape(-1, 1)
    opt.tell(x0, y0)
    assert opt.tr_obs_count == 2
    _, _, best = run_ask_tell_cycle(opt, rng, num_arms=2, obj_fn=obj_fn, num_cycles=3)
    assert opt.tr_obs_count == 8
    assert best >= -1.0


def run_ask_tell_cycle(opt, rng, num_arms: int, obj_fn, num_cycles: int):
    """Run num_cycles ask/tell cycles, return xs, ys, and final best y."""
    xs, ys = [], []
    for _ in range(num_cycles):
        x = opt.ask(num_arms=num_arms)
        y = obj_fn(x)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        opt.tell(x, y)
        xs.append(x)
        ys.append(y)
    all_y = np.concatenate(ys, axis=0)
    best_y = float(np.max(all_y))
    return xs, ys, best_y


def check_opt_contract(opt, bounds: np.ndarray):
    """Assert optimizer satisfies basic contract: ask shape, bounds, telemetry."""
    x = opt.ask(num_arms=3)
    assert isinstance(x, np.ndarray)
    assert x.shape == (3, bounds.shape[0])
    assert np.all(x >= bounds[:, 0])
    assert np.all(x <= bounds[:, 1])
    t = opt.telemetry()
    assert hasattr(t, "dt_fit") and hasattr(t, "dt_gen")
    assert hasattr(t, "dt_sel") and hasattr(t, "dt_tell")
    return x


def get_rust_optimizer(bounds, config, seed: int):
    """Create optimizer using Rust backend (no fallback)."""
    rng = np.random.default_rng(seed)
    assert is_rust_supported_config(config)
    return create_optimizer(bounds=bounds, config=config, rng=rng)


def get_python_optimizer(bounds, config, seed: int):
    """Create optimizer using Python backend (bypass Rust routing)."""
    from unittest.mock import patch

    import enn.turbo.rust_optimizer as ro

    rng = np.random.default_rng(seed)
    with patch.object(ro, "is_rust_supported_config", return_value=False):
        return create_optimizer(bounds=bounds, config=config, rng=rng)
