"""Parity tests: TuRBO-ENN Rust vs Python optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.config import (
    AcqType,
    ENNFitConfig,
    ENNSurrogateConfig,
    turbo_enn_config,
)

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def _rust_enn_config(enn=None, num_init=6, **kwargs):
    if enn is None:
        enn = ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10))
    return turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=enn,
        num_init=num_init,
        **kwargs,
    )


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def test_optimizer_enn_contract_and_shape():
    """Rust TuRBO-ENN: ask returns correct shape, candidates in bounds."""
    from .optimizer_parity_helpers import check_opt_contract, get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = _rust_enn_config(
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)), num_init=4
    )
    opt = get_rust_optimizer(bounds, config, seed=7)
    check_opt_contract(opt, bounds)


def test_optimizer_enn_ask_tell_state():
    """Rust TuRBO-ENN: tr_obs_count increases after tell."""
    from .optimizer_parity_helpers import assert_rust_optimizer_tr_obs_after_cycles

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = _rust_enn_config(num_init=4)
    assert_rust_optimizer_tr_obs_after_cycles(
        bounds, config, opt_seed=11, cycle_rng_seed=11, obj_fn=_obj
    )


def test_optimizer_enn_convergence_tolerance():
    """Rust TuRBO-ENN: converges toward optimum on simple sphere (tolerance parity)."""
    from .optimizer_parity_helpers import get_rust_optimizer, run_ask_tell_cycle

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = _rust_enn_config(
        enn=ENNSurrogateConfig(k=5, fit=ENNFitConfig(num_fit_samples=10)), num_init=6
    )
    opt = get_rust_optimizer(bounds, config, seed=19)
    rng = np.random.default_rng(19)
    _, _, best = run_ask_tell_cycle(opt, rng, num_arms=4, obj_fn=_obj, num_cycles=8)
    assert best >= -0.1
