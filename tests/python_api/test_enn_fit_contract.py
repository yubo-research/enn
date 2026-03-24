"""Contract tests for enn_fit function."""

from __future__ import annotations

import inspect

import numpy as np

from enn import EpistemicNearestNeighbors, enn_fit
from enn.enn.enn_params import ENNParams


class TestEnnFitContract:
    """API contract tests for enn_fit."""

    def test_enn_fit_signature(self):
        sig = inspect.signature(enn_fit)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "k" in params
        assert "num_fit_candidates" in params
        assert "num_fit_samples" in params
        assert "rng" in params
        assert "params_warm_start" in params
        assert "infer_aleatoric_variance_scale" in params

    def test_enn_fit_returns_en_params(self):
        train_x = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=float,
        )
        train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
        model = EpistemicNearestNeighbors(train_x, train_y, scale_x=False)
        rng = np.random.default_rng(42)

        result = enn_fit(
            model,
            k=2,
            num_fit_candidates=3,
            num_fit_samples=2,
            rng=rng,
        )
        assert isinstance(result, ENNParams)
        assert result.k_num_neighbors == 2
        assert result.epistemic_variance_scale >= 0.0
        assert result.aleatoric_variance_scale >= 0.0
