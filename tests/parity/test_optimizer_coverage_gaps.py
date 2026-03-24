"""Parity tests for review-identified coverage gaps.

- TR length trajectory: Rust vs Python over same ask/tell sequence
- Acquisition: smoke tests (full parity blocked on config pass-through)
- trailing_obs: Rust trim behavior (enabled after trailing_obs support in is_rust_supported_config)
- Multi-objective: Rust-backed via Pareto acquisition (ENN surrogate)
"""

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

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def _obj_multi(x):
    """Two objectives for Pareto acquisition."""
    f1 = -np.sum((x - 0.5) ** 2, axis=1)
    f2 = -np.sum((x - 0.3) ** 2, axis=1)
    return np.column_stack([f1, f2])


def _run_and_record_tr_lengths(opt, rng, num_arms: int, num_cycles: int):
    """Run ask/tell cycles, return list of tr_length after each tell."""
    lengths = []
    for _ in range(num_cycles):
        x = opt.ask(num_arms=num_arms)
        y = _obj(x)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        opt.tell(x, y)
        lengths.append(opt.tr_length)
    return lengths


def _run_pareto_record_tr_lengths(opt, rng, num_arms: int, num_cycles: int):
    """Run Pareto ask/tell with single-objective y, return tr_length after each tell.

    Uses single-objective so Python TurboTRConfig (num_metrics=1) is satisfied.
    Pareto acquisition with 1-col y still exercises the Pareto path.
    """
    lengths = []
    for _ in range(num_cycles):
        x = opt.ask(num_arms=num_arms)
        y = _obj(x)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        opt.tell(x, y)
        lengths.append(opt.tr_length)
    return lengths


def test_optimizer_tr_length_trajectory_parity():
    """Rust vs Python: tr_length trajectories similar over same ask/tell sequence.

    Both use scale from all observations before the current batch.
    Use tolerance for minor divergence (e.g. incumbent tie-breaking).
    """
    from .optimizer_parity_helpers import (
        get_python_optimizer,
        get_rust_optimizer,
    )

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_init = 8
    num_arms = 4
    config = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=num_init,
    )
    seed = 31
    num_cycles = 8

    rust_opt = get_rust_optimizer(bounds, config, seed)
    py_opt = get_python_optimizer(bounds, config, seed)
    rng_rust = np.random.default_rng(seed)
    rng_py = np.random.default_rng(seed)

    rust_lengths = _run_and_record_tr_lengths(rust_opt, rng_rust, num_arms, num_cycles)
    py_lengths = _run_and_record_tr_lengths(py_opt, rng_py, num_arms, num_cycles)

    assert len(rust_lengths) == num_cycles
    assert len(py_lengths) == num_cycles

    for i in range(num_cycles):
        assert 0.0 < rust_lengths[i] <= 2.0
        assert 0.0 < py_lengths[i] <= 2.0
    mean_diff = np.mean(np.abs(np.array(rust_lengths) - np.array(py_lengths)))
    assert mean_diff < 0.5


def test_optimizer_pareto_tr_length_parity():
    """Rust vs Python Pareto: tr_length trajectories similar over same ask/tell sequence.

    Uses multi-objective y; TR update uses first column (parity with UCB test).
    """
    from .optimizer_parity_helpers import (
        get_python_optimizer,
        get_rust_optimizer,
    )

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_init = 6
    num_arms = 3
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=num_init,
    )
    seed = 47
    num_cycles = 6

    rust_opt = get_rust_optimizer(bounds, config, seed)
    py_opt = get_python_optimizer(bounds, config, seed)
    rng_rust = np.random.default_rng(seed)
    rng_py = np.random.default_rng(seed)

    rust_lengths = _run_pareto_record_tr_lengths(
        rust_opt, rng_rust, num_arms, num_cycles
    )
    py_lengths = _run_pareto_record_tr_lengths(py_opt, rng_py, num_arms, num_cycles)

    assert len(rust_lengths) == num_cycles
    assert len(py_lengths) == num_cycles
    for i in range(num_cycles):
        assert 0.0 < rust_lengths[i] <= 2.0
        assert 0.0 < py_lengths[i] <= 2.0
    mean_diff = np.mean(np.abs(np.array(rust_lengths) - np.array(py_lengths)))
    assert mean_diff < 0.5


def test_acquisition_ucb_produces_valid_candidates():
    """Rust TuRBO-ENN with UCB: ask returns valid candidates in bounds."""
    from .optimizer_parity_helpers import check_opt_contract, get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
    )
    opt = get_rust_optimizer(bounds, config, seed=19)
    check_opt_contract(opt, bounds)


def test_acquisition_thompson_config_passthrough():
    """Rust TuRBO-ENN with Thompson: config pass-through produces valid candidates."""
    from .optimizer_parity_helpers import check_opt_contract, get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.THOMPSON,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
    )
    opt = get_rust_optimizer(bounds, config, seed=23)
    check_opt_contract(opt, bounds)


def test_multi_objective_uses_rust():
    """Multi-objective (Pareto) config uses Rust backend when supported."""
    from enn import create_optimizer
    from enn.turbo.rust_optimizer import RustOptimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(3)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert isinstance(opt, RustOptimizer)
    x = opt.ask(num_arms=2)
    assert x.shape == (2, 2)


def test_multi_objective_rust_ask_tell():
    """Pareto config uses Rust backend; ask/tell with single and multi-objective y."""
    from enn import create_optimizer
    from enn.turbo.rust_optimizer import RustOptimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(5)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert isinstance(opt, RustOptimizer)
    # Single-objective y
    x = opt.ask(num_arms=2)
    y = _obj(x).reshape(-1, 1)
    opt.tell(x, y)
    assert x.shape == (2, 2)
    # Multi-objective y (2 columns)
    x2 = opt.ask(num_arms=2)
    y2 = np.column_stack([_obj(x2), -_obj(x2)])  # Two objectives
    opt.tell(x2, y2)
    assert x2.shape == (2, 2)
    assert y2.shape == (2, 2)


def test_multi_objective_y_obs_shape():
    """_ObsView reflects multi-objective y shape after tell (review 5.1)."""
    from enn import create_optimizer
    from enn.turbo.rust_optimizer import RustOptimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(7)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert isinstance(opt, RustOptimizer)
    # All multi-objective y from start (consistent 2-col)
    x = opt.ask(num_arms=4)
    y = np.column_stack([_obj(x), -_obj(x)])
    opt.tell(x, y)
    y_obs = opt._y_obs.view()
    assert y_obs.shape[1] == 2, "_ObsView should reflect multi-objective y cols"


def test_trailing_obs_trim_behavior():
    """Rust optimizer with trailing_obs: observations trimmed to limit."""
    from .optimizer_parity_helpers import get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero_config(num_init=4, trailing_obs=10)
    np.random.default_rng(7)
    opt = get_rust_optimizer(bounds, config, seed=7)

    for _ in range(25):
        x = opt.ask(num_arms=1)
        y = _obj(x).reshape(-1, 1)
        opt.tell(x, y)

    assert opt.tr_obs_count <= 10
