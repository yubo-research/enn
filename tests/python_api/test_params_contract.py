"""Contract tests for ENNParams and PosteriorFlags."""

from __future__ import annotations

import inspect

import pytest

from enn.enn.enn_params import ENNParams, PosteriorFlags


class TestENNParamsContract:
    """API contract tests for ENNParams."""

    def test_exists_and_is_class(self):
        from enn.enn.enn_params import ENNParams

        assert inspect.isclass(ENNParams)

    def test_signature_contract(self):
        sig = inspect.signature(ENNParams)
        params = list(sig.parameters.keys())
        assert "k_num_neighbors" in params
        assert "epistemic_variance_scale" in params
        assert "aleatoric_variance_scale" in params

    def test_valid_params_constructible(self):
        p = ENNParams(
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        )
        assert p.k_num_neighbors == 2
        assert p.epistemic_variance_scale == 1.0
        assert p.aleatoric_variance_scale == 0.1

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k_num_neighbors"):
            ENNParams(
                k_num_neighbors=0,
                epistemic_variance_scale=1.0,
                aleatoric_variance_scale=0.0,
            )

    def test_invalid_epistemic_raises(self):
        with pytest.raises(ValueError, match="epistemic_variance_scale"):
            ENNParams(
                k_num_neighbors=2,
                epistemic_variance_scale=-1.0,
                aleatoric_variance_scale=0.0,
            )

    def test_invalid_aleatoric_raises(self):
        with pytest.raises(ValueError, match="aleatoric_variance_scale"):
            ENNParams(
                k_num_neighbors=2,
                epistemic_variance_scale=1.0,
                aleatoric_variance_scale=-0.1,
            )

    def test_is_frozen(self):
        p = ENNParams(
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            p.k_num_neighbors = 3


class TestPosteriorFlagsContract:
    """API contract tests for PosteriorFlags."""

    def test_exists_and_is_class(self):
        from enn.enn.enn_params import PosteriorFlags

        assert inspect.isclass(PosteriorFlags)

    def test_default_values(self):
        flags = PosteriorFlags()
        assert flags.exclude_nearest is False
        assert flags.observation_noise is False
        assert flags.tie_break_neighbors is True

    def test_explicit_values(self):
        flags = PosteriorFlags(
            exclude_nearest=True,
            observation_noise=True,
            tie_break_neighbors=False,
        )
        assert flags.exclude_nearest is True
        assert flags.observation_noise is True
        assert flags.tie_break_neighbors is False

    def test_signature_contract(self):
        sig = inspect.signature(PosteriorFlags)
        params = list(sig.parameters.keys())
        assert "exclude_nearest" in params
        assert "observation_noise" in params
        assert "tie_break_neighbors" in params
