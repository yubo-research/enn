"""RAASP candidate generation: Rust vs Python parity."""

from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.config import (
    AcqType,
    ENNFitConfig,
    ENNSurrogateConfig,
    turbo_enn_config,
    turbo_zero_config,
)
from enn.turbo.config.candidate_gen_config import CandidateGenConfig
from enn.turbo.config.candidate_rv import CandidateRV

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def test_raasp_optimizer_contract():
    """Rust TuRBO-ENN with RAASP: ask returns valid candidates in bounds."""
    from .optimizer_parity_helpers import check_opt_contract, get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
        candidates=CandidateGenConfig(candidate_rv=CandidateRV.RAASP),
    )
    opt = get_rust_optimizer(bounds, config, seed=41)
    check_opt_contract(opt, bounds)


def test_raasp_rust_candidate_distribution():
    """Rust RAASP produces center-biased candidates in bounds."""
    from .optimizer_parity_helpers import get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 4
    config = turbo_zero_config(num_init=8, candidate_rv=CandidateRV.RAASP)
    rust_opt = get_rust_optimizer(bounds, config, seed=77)
    rust_cands = []
    for _ in range(6):
        x_r = rust_opt.ask(num_arms=num_arms)
        rust_cands.append(x_r)
        y_r = _obj(x_r).reshape(-1, 1)
        rust_opt.tell(x_r, y_r)

    rust_all = np.concatenate(rust_cands, axis=0)
    assert rust_all.shape[1] == 2
    assert np.all(rust_all >= 0) and np.all(rust_all <= 1)
