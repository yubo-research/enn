"""Parity tests: TuRBO-ZERO Rust vs Python optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.config import turbo_zero_config

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def test_optimizer_zero_contract_and_shape():
    """Rust TuRBO-ZERO: ask returns correct shape, candidates in bounds."""
    from .optimizer_parity_helpers import check_opt_contract, get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero_config(num_init=4)
    opt = get_rust_optimizer(bounds, config, seed=13)
    check_opt_contract(opt, bounds)


def test_optimizer_zero_ask_tell_state():
    """Rust TuRBO-ZERO: tr_obs_count increases after tell."""
    from .optimizer_parity_helpers import get_rust_optimizer, run_ask_tell_cycle

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero_config(num_init=4)
    rng = np.random.default_rng(17)
    opt = get_rust_optimizer(bounds, config, seed=17)
    x0 = opt.ask(num_arms=2)
    assert opt.tr_obs_count == 0
    y0 = _obj(x0).reshape(-1, 1)
    opt.tell(x0, y0)
    assert opt.tr_obs_count == 2
    _, _, best = run_ask_tell_cycle(opt, rng, num_arms=2, obj_fn=_obj, num_cycles=3)
    assert opt.tr_obs_count == 8
    assert best >= -1.0
