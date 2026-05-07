"""MORBO on a separable 2D, 2-metric toy problem with mismatched x/y scales."""

from __future__ import annotations

import numpy as np

from enn import create_optimizer
from enn.turbo import turbo_utils
from enn.turbo.config import (
    AcqType,
    CandidateGenConfig,
    ENNFitConfig,
    ENNSurrogateConfig,
    MorboTRConfig,
    MultiObjectiveConfig,
    turbo_enn_config,
)


def _separable_unimodal_objective(x: np.ndarray) -> np.ndarray:
    """Two independent concave quadratics (unimodal on R) restricted to the box.

    x_1 controls y_1 only; x_2 controls y_2 only.  Deliberately different x ranges,
    peak locations, and y magnitudes so normalization / scalarization paths see
    heterogeneous scales.
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    y1 = 500_000.0 - 8.0 * (x1 - 120.0) ** 2
    y2 = 12.5 - 110.0 * (x2 - 0.91) ** 2
    return np.stack([y1, y2], axis=1)


def test_morbo_finds_joint_near_optimum_on_separable_unimodal_problem():
    # x_1 spans 900 units centered at 150; x_2 spans 0.8 on [0.2, 1.0] (different scale).
    bounds = np.array([[-300.0, 600.0], [0.2, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(
            k=4,
            fit=ENNFitConfig(num_fit_samples=4, num_fit_candidates=8),
        ),
        trust_region=MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2)),
        num_init=8,
        candidates=CandidateGenConfig(num_candidates=lambda *, num_dim, num_arms: 64),
        acq_type=AcqType.THOMPSON,
    )
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    num_rounds = 35
    num_arms = 4
    for _ in range(num_rounds):
        x = opt.ask(num_arms=num_arms)
        opt.tell(x, _separable_unimodal_objective(x))

    x_unit = np.asarray(opt._x_obs.view(), dtype=float)
    x_phys = turbo_utils.from_unit(x_unit, bounds)
    y = _separable_unimodal_objective(x_phys)

    lb, ub = bounds[:, 0], bounds[:, 1]
    assert np.all(np.isfinite(x_phys)) and np.all(np.isfinite(y))
    assert np.all(x_phys >= lb) and np.all(x_phys <= ub)
    assert y.shape[1] == 2

    # Each objective was pushed near its global peak on some trial (not only a scalarized compromise).
    assert float(np.max(y[:, 0])) >= 499_000.0
    assert float(np.max(y[:, 1])) >= 11.0

    # Unique maximizer at (120, 0.91); require a single evaluated point close on both metrics.
    joint = (y[:, 0] >= 499_000.0) & (y[:, 1] >= 11.0)
    assert np.any(joint), "expected at least one near-joint-optimum evaluation"
