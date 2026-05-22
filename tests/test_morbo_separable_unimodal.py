"""MORBO on a separable 2D, 2-metric toy problem with mismatched x/y scales."""

from __future__ import annotations

import numpy as np

from enn import create_optimizer
from enn.turbo.config import (
    AcqType,
    CandidateGenConfig,
    ENNFitConfig,
    ENNSurrogateConfig,
    MorboTRConfig,
    MultiObjectiveConfig,
    turbo_enn_config,
)

from tests.morbo_objectives import (
    separable_unimodal_objective as _separable_unimodal_objective,
)


def test_morbo_finds_joint_near_optimum_on_separable_unimodal_problem():
    bounds = np.array([[-300.0, 600.0], [0.2, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(
            k=4,
            fit=ENNFitConfig(num_fit_samples=4, num_fit_candidates=8),
        ),
        trust_region=MorboTRConfig(
            multi_objective=MultiObjectiveConfig(num_metrics=2),
        ),
        num_init=8,
        candidates=CandidateGenConfig(num_candidates=64),
        acq_type=AcqType.THOMPSON,
    )
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    num_arms = 4
    while opt.init_progress is not None:
        x = opt.ask(num_arms=num_arms)
        y = _separable_unimodal_objective(x)
        opt.tell(x, y)
    best_y = np.array([-np.inf, -np.inf], dtype=float)
    rounds = 0
    while rounds < 35:
        x = opt.ask(num_arms=num_arms)
        y = _separable_unimodal_objective(x)
        opt.tell(x, y)
        best_y = np.maximum(best_y, y.max(axis=0))
        rounds += 1
    assert np.all(np.isfinite(best_y))
    assert float(best_y[0]) >= 499_000.0
    assert float(best_y[1]) >= 11.0
