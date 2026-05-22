"""Morbo trust region with Pareto acquisition on the Rust optimizer path."""

from __future__ import annotations

import numpy as np
import pytest

from enn import create_optimizer, turbo_enn_config
from enn.turbo.config import (
    AcqType,
    CandidateGenConfig,
    ENNFitConfig,
    ENNSurrogateConfig,
    MorboTRConfig,
    MultiObjectiveConfig,
)
from enn.turbo.rust_optimizer import RustOptimizer

from tests.morbo_objectives import separable_unimodal_objective

try:
    from enn import _rust  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def test_morbo_pareto_rust_post_init_contracts():
    bounds = np.array([[-300.0, 600.0], [0.2, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(
            k=4,
            fit=ENNFitConfig(num_fit_samples=4, num_fit_candidates=8),
        ),
        trust_region=MorboTRConfig(
            multi_objective=MultiObjectiveConfig(num_metrics=2),
            noise_aware=True,
        ),
        num_init=8,
        candidates=CandidateGenConfig(num_candidates=64),
    )
    rng = np.random.default_rng(43)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert isinstance(opt, RustOptimizer)
    num_arms = 4
    while opt.init_progress is not None:
        x = opt.ask(num_arms=num_arms)
        assert x.shape == (num_arms, 2)
        y = separable_unimodal_objective(x)
        assert y.shape == (num_arms, 2)
        opt.tell(x, y)
    x = opt.ask(num_arms=num_arms)
    y = separable_unimodal_objective(x)
    opt.tell(x, y)
    assert opt.telemetry().num_candidates == 64
    assert int(opt.tr_obs_count) >= num_arms


def test_morbo_pareto_rust_improves_both_objectives():
    bounds = np.array([[-300.0, 600.0], [0.2, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(
            k=4,
            fit=ENNFitConfig(num_fit_samples=4, num_fit_candidates=8),
        ),
        trust_region=MorboTRConfig(
            multi_objective=MultiObjectiveConfig(num_metrics=2),
            noise_aware=True,
        ),
        num_init=8,
        candidates=CandidateGenConfig(num_candidates=64),
    )
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert isinstance(opt, RustOptimizer)
    best_y = np.array([-np.inf, -np.inf], dtype=float)
    num_rounds = 35
    num_arms = 4
    for _ in range(num_rounds):
        x = opt.ask(num_arms=num_arms)
        assert x.shape == (num_arms, 2)
        y = separable_unimodal_objective(x)
        opt.tell(x, y)
        best_y = np.maximum(best_y, y.max(axis=0))
        assert np.isfinite(opt.tr_length)
    assert float(best_y[0]) >= 499_000.0
    assert float(best_y[1]) >= 11.0
