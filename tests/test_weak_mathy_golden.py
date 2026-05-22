"""Golden and parity tests for weak_tests.md mathy regions."""

from __future__ import annotations

import numpy as np

from enn.turbo.python_fallback.components.chebyshev_incumbent_selector import (
    ChebyshevIncumbentSelector,
)


def test_chebyshev_scalarize_batch_normalized_golden():
    sel = ChebyshevIncumbentSelector(num_metrics=2, noise_aware=False, alpha=0.1)
    sel._weights = np.array([0.5, 0.5])
    y = np.array([[1.0, 2.0], [1.0, 3.0]])
    scores = sel._scalarize(y)
    assert np.allclose(scores[0], 0.025, rtol=0.0, atol=1e-12)
    assert np.allclose(scores[1], 0.325, rtol=0.0, atol=1e-12)


def test_subsample_loglik_fixed_seed_reproducible():
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import subsample_loglik
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 2))
    y = (x @ np.array([1.0, -0.5]) + 0.05 * rng.standard_normal(20)).reshape(-1, 1)
    model = EpistemicNearestNeighbors(x, y, 0.01 * np.ones_like(y))
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    ll1 = subsample_loglik(
        model, x, y[:, 0], paramss=[params], P=10, rng=np.random.default_rng(11)
    )[0]
    ll2 = subsample_loglik(
        model, x, y[:, 0], paramss=[params], P=10, rng=np.random.default_rng(11)
    )[0]
    assert ll1 == ll2
    assert np.isfinite(ll1)
