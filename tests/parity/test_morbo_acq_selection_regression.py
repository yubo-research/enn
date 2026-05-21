"""Failing regressions for Morbo acquisition RNG parity (Rust vs Python reference)."""

from __future__ import annotations

import numpy as np

from enn.turbo.config.morbo_tr_config import MorboTRConfig, MultiObjectiveConfig
from enn.turbo.python_fallback.components.thompson_acq_optimizer import (
    ThompsonAcqOptimizer,
)
from enn.turbo.python_fallback.components.ucb_acq_optimizer import UCBAcqOptimizer
from enn.turbo.python_fallback.morbo_trust_region import MorboTrustRegion


class _TieSurrogate:
    def sample(
        self, x_cand: np.ndarray, num_arms: int, rng: np.random.Generator
    ) -> np.ndarray:
        n = len(x_cand)
        m = 2
        return np.ones((num_arms, n, m), dtype=float)

    def predict(self, x_cand: np.ndarray):
        from enn.turbo.python_fallback.components.posterior_result import (
            PosteriorResult,
        )

        n = len(x_cand)
        return PosteriorResult(
            mu=np.ones((n, 2), dtype=float),
            sigma=np.zeros((n, 2), dtype=float),
        )


def _morbo_tr(rng: np.random.Generator) -> MorboTrustRegion:
    cfg = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    tr = MorboTrustRegion(cfg, num_dim=2, rng=rng)
    tr.validate_request(2)
    y_obs = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]], dtype=float)
    tr.update(y_obs, tr.get_incumbent_value(y_obs, rng))
    return tr


def test_morbo_thompson_python_breaks_ties_with_rng():
    x_cand = np.linspace(0.0, 1.0, 8 * 2).reshape(8, 2)
    picks: list[tuple[tuple[float, ...], ...]] = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        tr = _morbo_tr(rng)
        sel = ThompsonAcqOptimizer().select(
            x_cand, 2, _TieSurrogate(), rng, tr_state=tr
        )
        picks.append(tuple(tuple(row) for row in sel))
    assert len(set(picks)) > 1


def test_morbo_ucb_python_breaks_ties_with_rng():
    x_cand = np.linspace(0.0, 1.0, 8 * 2).reshape(8, 2)
    picks: list[tuple[tuple[float, ...], ...]] = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        tr = _morbo_tr(rng)
        sel = UCBAcqOptimizer(beta=1.0).select(
            x_cand, 2, _TieSurrogate(), rng, tr_state=tr
        )
        picks.append(tuple(tuple(row) for row in sel))
    assert len(set(picks)) > 1
