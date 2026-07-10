"""Contract tests for EpistemicNearestNeighbors posterior methods."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from enn import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


@pytest.fixture
def simple_model():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    return EpistemicNearestNeighbors(train_x, train_y, scale_x=False)


@pytest.fixture
def query():
    return np.array([[0.5, 0.5]], dtype=float)


class TestPosteriorContract:
    """API contract tests for posterior."""

    def test_posterior_exists_and_callable(self):
        assert hasattr(EpistemicNearestNeighbors, "posterior")
        assert callable(getattr(EpistemicNearestNeighbors, "posterior"))

    def test_posterior_returns_mu_se(self, simple_model, query):
        params = ENNParams(
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        )
        flags = PosteriorFlags()
        out = simple_model.posterior(query, params=params, flags=flags)
        assert hasattr(out, "mu")
        assert hasattr(out, "se")
        assert hasattr(out, "se_epi")
        assert hasattr(out, "se_ale")
        assert out.mu.shape == (1, 1)
        assert out.se.shape == (1, 1)
        assert out.se_epi.shape == (1, 1)
        assert out.se_ale.shape == (1, 1)
        assert np.all(np.isfinite(out.mu))
        assert np.all(np.isfinite(out.se))


class TestBatchPosteriorContract:
    """API contract tests for batch_posterior."""

    def test_batch_posterior_exists_and_callable(self):
        assert hasattr(EpistemicNearestNeighbors, "batch_posterior")
        assert callable(getattr(EpistemicNearestNeighbors, "batch_posterior"))

    def test_batch_posterior_signature(self):
        sig = inspect.signature(EpistemicNearestNeighbors.batch_posterior)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "x" in params
        assert "paramss" in params
        assert "flags" in params

    def test_batch_posterior_returns_envelope_with_mu_se(self, simple_model, query):
        params1 = ENNParams(
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        )
        params2 = ENNParams(
            k_num_neighbors=3,
            epistemic_variance_scale=2.0,
            aleatoric_variance_scale=0.2,
        )
        out = simple_model.batch_posterior(query, paramss=[params1, params2])
        assert hasattr(out, "mu")
        assert hasattr(out, "se")
        assert hasattr(out, "se_epi")
        assert hasattr(out, "se_ale")
        assert out.mu.shape == (2, 1, 1)
        assert out.se.shape == (2, 1, 1)
        assert out.se_epi.shape == (2, 1, 1)
        assert out.se_ale.shape == (2, 1, 1)

    def test_batch_posterior_empty_paramss_raises(self, simple_model, query):
        with pytest.raises(ValueError, match="paramss must be non-empty"):
            simple_model.batch_posterior(query, paramss=[])


class TestConditionalPosteriorContract:
    """API contract tests for conditional_posterior."""

    def test_conditional_posterior_exists_and_callable(self):
        assert hasattr(EpistemicNearestNeighbors, "conditional_posterior")
        assert callable(getattr(EpistemicNearestNeighbors, "conditional_posterior"))

    def test_conditional_posterior_signature(self):
        sig = inspect.signature(EpistemicNearestNeighbors.conditional_posterior)
        params = list(sig.parameters.keys())
        assert "x_whatif" in params
        assert "y_whatif" in params
        assert "x" in params
        assert "params" in params

    def test_conditional_posterior_empty_whatif_matches_posterior(
        self, simple_model, query
    ):
        params = ENNParams(
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        )
        flags = PosteriorFlags()
        x_whatif = np.zeros((0, 2), dtype=float)
        y_whatif = np.zeros((0, 1), dtype=float)

        post = simple_model.posterior(query, params=params, flags=flags)
        cond_post = simple_model.conditional_posterior(
            x_whatif, y_whatif, query, params=params, flags=flags
        )

        np.testing.assert_allclose(post.mu, cond_post.mu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(post.se, cond_post.se, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            post.se_epi, cond_post.se_epi, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            post.se_ale, cond_post.se_ale, rtol=1e-12, atol=1e-12
        )

    def test_conditional_posterior_with_whatif_returns_mu_se(self, simple_model, query):
        x_whatif = np.array([[0.5, 0.5]], dtype=float)
        y_whatif = np.array([[1.5]], dtype=float)
        params = ENNParams(
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        )
        flags = PosteriorFlags()

        out = simple_model.conditional_posterior(
            x_whatif, y_whatif, query, params=params, flags=flags
        )
        assert hasattr(out, "mu")
        assert hasattr(out, "se")
        assert hasattr(out, "se_epi")
        assert hasattr(out, "se_ale")
        assert out.mu.shape == (1, 1)
        assert out.se.shape == (1, 1)
        assert out.se_epi.shape == (1, 1)
        assert out.se_ale.shape == (1, 1)


class TestPosteriorFunctionDrawContract:
    """API contract tests for posterior_function_draw."""

    def test_posterior_function_draw_exists_and_callable(self):
        assert hasattr(EpistemicNearestNeighbors, "posterior_function_draw")
        assert callable(getattr(EpistemicNearestNeighbors, "posterior_function_draw"))

    def test_posterior_function_draw_signature(self):
        sig = inspect.signature(EpistemicNearestNeighbors.posterior_function_draw)
        params = list(sig.parameters.keys())
        assert "x" in params
        assert "params" in params
        assert "function_seeds" in params

    def test_posterior_function_draw_returns_tuple_of_draws_and_idx(
        self, simple_model, query
    ):
        params = ENNParams(
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        )
        flags = PosteriorFlags()
        seeds = [42, 43]

        draws, idx = simple_model.posterior_function_draw(
            query, params=params, function_seeds=seeds, flags=flags
        )

        assert isinstance(draws, np.ndarray)
        assert draws.shape == (2, 1, 1)
        assert isinstance(idx, (list, np.ndarray))
        assert len(idx) == 1

    def test_conditional_posterior_function_draw_exists_and_callable(self):
        assert hasattr(EpistemicNearestNeighbors, "conditional_posterior_function_draw")
        assert callable(
            getattr(EpistemicNearestNeighbors, "conditional_posterior_function_draw")
        )

    def test_conditional_posterior_function_draw_returns_draws_and_idx(
        self, simple_model, query
    ):
        x_whatif = np.array([[0.5, 0.5]], dtype=float)
        y_whatif = np.array([[1.5]], dtype=float)
        params = ENNParams(
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        )
        flags = PosteriorFlags()
        seeds = [42]

        draws, idx = simple_model.conditional_posterior_function_draw(
            x_whatif, y_whatif, query, params=params, function_seeds=seeds, flags=flags
        )

        assert isinstance(draws, np.ndarray)
        assert draws.shape == (1, 1, 1)
        assert isinstance(idx, (list, np.ndarray))
