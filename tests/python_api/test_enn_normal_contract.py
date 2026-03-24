"""Contract tests for ENNNormal (posterior result type)."""

from __future__ import annotations

import inspect

import numpy as np

from enn import EpistemicNearestNeighbors
from enn.enn.enn_normal import ENNNormal
from enn.enn.enn_params import ENNParams, PosteriorFlags


class TestENNNormalContract:
    """API contract tests for ENNNormal return type."""

    def test_exists_and_is_class(self):
        from enn.enn.enn_normal import ENNNormal

        assert inspect.isclass(ENNNormal)

    def test_has_mu_se_idx_attrs(self):
        mu = np.array([[1.0]], dtype=float)
        se = np.array([[0.2]], dtype=float)
        obj = ENNNormal(mu=mu, se=se)
        assert hasattr(obj, "mu")
        assert hasattr(obj, "se")
        assert hasattr(obj, "idx")
        assert obj.mu is mu
        assert obj.se is se
        assert obj.idx is None

    def test_idx_optional(self):
        mu = np.array([[1.0]], dtype=float)
        se = np.array([[0.2]], dtype=float)
        idx = np.array([[0, 1]], dtype=int)
        obj = ENNNormal(mu=mu, se=se, idx=idx)
        assert obj.idx is idx

    def test_posterior_returns_enn_normal(self):
        train_x = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
        )
        train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
        model = EpistemicNearestNeighbors(train_x, train_y, scale_x=False)
        params = ENNParams(
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        )
        flags = PosteriorFlags()
        query = np.array([[0.5, 0.5]], dtype=float)

        out = model.posterior(query, params=params, flags=flags)
        assert isinstance(out, ENNNormal)
        assert out.mu.shape == (1, 1)
        assert out.se.shape == (1, 1)
        assert np.all(np.isfinite(out.mu))
        assert np.all(np.isfinite(out.se))

    def test_sample_method_exists(self):
        mu = np.array([[1.0]], dtype=float)
        se = np.array([[0.2]], dtype=float)
        obj = ENNNormal(mu=mu, se=se)
        assert hasattr(obj, "sample")
        assert callable(obj.sample)

    def test_sample_signature(self):
        sig = inspect.signature(ENNNormal.sample)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "num_samples" in params
        assert "rng" in params
        assert "clip" in params

    def test_sample_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        mu = np.array([[1.0, 2.0]], dtype=float)
        se = np.array([[0.1, 0.2]], dtype=float)
        obj = ENNNormal(mu=mu, se=se)
        samples = obj.sample(num_samples=10, rng=rng)
        # shape is (*se.shape, num_samples)
        assert samples.shape == (1, 2, 10)
        assert np.all(np.isfinite(samples))
