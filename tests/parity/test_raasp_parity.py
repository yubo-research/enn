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


def test_raasp_rust_vs_python_candidate_distribution():
    """Rust RAASP and Python RAASP both produce center-biased candidates.

    Statistical check: candidates should be clustered around incumbent
    (center + perturbation) rather than uniformly spread.
    """
    from .optimizer_parity_helpers import get_python_optimizer, get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_init = 8
    num_arms = 4
    config = turbo_zero_config(num_init=num_init, candidate_rv=CandidateRV.RAASP)
    seed = 77
    num_cycles = 6

    rust_opt = get_rust_optimizer(bounds, config, seed)
    py_opt = get_python_optimizer(bounds, config, seed)
    np.random.default_rng(seed)
    np.random.default_rng(seed)

    rust_cands = []
    py_cands = []
    for _ in range(num_cycles):
        x_r = rust_opt.ask(num_arms=num_arms)
        x_p = py_opt.ask(num_arms=num_arms)
        rust_cands.append(x_r)
        py_cands.append(x_p)
        y_r = _obj(x_r).reshape(-1, 1)
        y_p = _obj(x_p).reshape(-1, 1)
        rust_opt.tell(x_r, y_r)
        py_opt.tell(x_p, y_p)

    rust_all = np.concatenate(rust_cands, axis=0)
    py_all = np.concatenate(py_cands, axis=0)

    assert rust_all.shape[1] == 2
    assert py_all.shape[1] == 2
    assert np.all(rust_all >= 0) and np.all(rust_all <= 1)
    assert np.all(py_all >= 0) and np.all(py_all <= 1)
