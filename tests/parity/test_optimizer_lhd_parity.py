"""Parity tests: LHD_ONLY Rust vs Python optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.config import lhd_only_config

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def test_optimizer_lhd_contract_and_shape():
    """Rust LHD_ONLY: ask returns correct shape, candidates in bounds."""
    from .optimizer_parity_helpers import check_opt_contract, get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = lhd_only_config(num_init=5)
    opt = get_rust_optimizer(bounds, config, seed=31)
    check_opt_contract(opt, bounds)


def test_optimizer_lhd_ask_tell_state():
    """Rust LHD_ONLY: tr_obs_count increases after tell."""
    from .optimizer_parity_helpers import assert_rust_optimizer_tr_obs_after_cycles

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = lhd_only_config(num_init=5)
    assert_rust_optimizer_tr_obs_after_cycles(
        bounds, config, opt_seed=37, cycle_rng_seed=37, obj_fn=_obj
    )
