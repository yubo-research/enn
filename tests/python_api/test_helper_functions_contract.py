from __future__ import annotations

import inspect

import numpy as np
import pytest


class TestHypervolumeContract:
    """API contract tests for hypervolume_2d_max function."""

    def test_function_exists_and_callable(self):
        """hypervolume_2d_max must exist and be callable."""
        from enn.turbo.hypervolume import hypervolume_2d_max

        assert callable(hypervolume_2d_max)

    def test_signature_contract(self):
        """Function signature must match contract: (y, ref_point) -> float."""
        from enn.turbo.hypervolume import hypervolume_2d_max

        sig = inspect.signature(hypervolume_2d_max)
        params = list(sig.parameters.keys())
        assert params == ["y", "ref_point"]
        # Note: return_annotation may be string 'float' due to future annotations
        assert str(sig.return_annotation) in ("float", "<class 'float'>")

    def test_valid_input_returns_float(self):
        """Valid 2D input returns non-negative float."""
        from enn.turbo.hypervolume import hypervolume_2d_max

        y = np.array([[1.0, 0.5], [0.5, 1.0]])
        ref = np.array([0.0, 0.0])
        result = hypervolume_2d_max(y, ref)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_empty_array_returns_zero(self):
        """Empty input array returns 0.0."""
        from enn.turbo.hypervolume import hypervolume_2d_max

        y = np.array([]).reshape(0, 2)
        ref = np.array([0.0, 0.0])
        result = hypervolume_2d_max(y, ref)
        assert result == 0.0

    def test_no_dominating_points_returns_zero(self):
        """When no points dominate ref_point, returns 0.0."""
        from enn.turbo.hypervolume import hypervolume_2d_max

        y = np.array([[-1.0, -1.0], [-0.5, -0.5]])  # All below ref
        ref = np.array([0.0, 0.0])
        result = hypervolume_2d_max(y, ref)
        assert result == 0.0

    def test_invalid_y_ndim_raises(self):
        """1D y array raises ValueError."""
        from enn.turbo.hypervolume import hypervolume_2d_max

        y = np.array([1.0, 0.5])
        ref = np.array([0.0, 0.0])
        with pytest.raises(ValueError):
            hypervolume_2d_max(y, ref)

    def test_invalid_y_shape_raises(self):
        """y with wrong second dimension raises ValueError."""
        from enn.turbo.hypervolume import hypervolume_2d_max

        y = np.array([[1.0, 0.5, 0.3]])  # 3D instead of 2D
        ref = np.array([0.0, 0.0])
        with pytest.raises(ValueError):
            hypervolume_2d_max(y, ref)

    def test_invalid_ref_shape_raises(self):
        """ref_point with wrong shape raises ValueError."""
        from enn.turbo.hypervolume import hypervolume_2d_max

        y = np.array([[1.0, 0.5]])
        ref = np.array([0.0])  # Wrong shape
        with pytest.raises(ValueError):
            hypervolume_2d_max(y, ref)


class TestEnnHashContract:
    """API contract tests for enn_hash RNG functions."""

    def test_normal_hash_batch_multi_seed_exists(self):
        """normal_hash_batch_multi_seed function exists."""
        from enn.enn.enn_hash import normal_hash_batch_multi_seed

        assert callable(normal_hash_batch_multi_seed)

    def test_normal_hash_batch_multi_seed_fast_exists(self):
        """normal_hash_batch_multi_seed_fast function exists."""
        from enn.enn.enn_hash import normal_hash_batch_multi_seed_fast

        assert callable(normal_hash_batch_multi_seed_fast)

    def test_hash_signature_contract(self):
        """Hash functions have signature (seeds, indices, num_metrics) -> array."""
        from enn.enn.enn_hash import (
            normal_hash_batch_multi_seed,
            normal_hash_batch_multi_seed_fast,
        )

        for fn in [normal_hash_batch_multi_seed, normal_hash_batch_multi_seed_fast]:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            assert params == ["function_seeds", "data_indices", "num_metrics"]

    def test_hash_determinism_contract(self):
        """Same inputs produce same outputs (determinism)."""
        from enn.enn.enn_hash import normal_hash_batch_multi_seed_fast

        seeds = np.array([42], dtype=np.int64)
        indices = np.array([[0, 1]], dtype=int)

        result1 = normal_hash_batch_multi_seed_fast(seeds, indices, num_metrics=2)
        result2 = normal_hash_batch_multi_seed_fast(seeds, indices, num_metrics=2)

        assert np.allclose(result1, result2)

    def test_hash_output_shape_contract(self):
        """Output shape is (num_seeds, *data_indices.shape, num_metrics)."""
        from enn.enn.enn_hash import normal_hash_batch_multi_seed_fast

        seeds = np.array([1, 2], dtype=np.int64)  # 2 seeds
        indices = np.array([[0, 1, 2], [3, 4, 5]])  # shape (2, 3)
        num_metrics = 4

        result = normal_hash_batch_multi_seed_fast(seeds, indices, num_metrics)

        assert result.shape == (2, 2, 3, 4)

    def test_hash_seed_sensitivity_contract(self):
        """Different seeds produce different outputs."""
        from enn.enn.enn_hash import normal_hash_batch_multi_seed_fast

        seeds1 = np.array([42], dtype=np.int64)
        seeds2 = np.array([99], dtype=np.int64)
        indices = np.array([[0, 1]], dtype=int)

        result1 = normal_hash_batch_multi_seed_fast(seeds1, indices, num_metrics=2)
        result2 = normal_hash_batch_multi_seed_fast(seeds2, indices, num_metrics=2)

        assert not np.allclose(result1, result2)

    def test_fast_hash_num_metrics_validation(self):
        """num_metrics <= 0 raises ValueError."""
        from enn.enn.enn_hash import normal_hash_batch_multi_seed_fast

        seeds = np.array([42], dtype=np.int64)
        indices = np.array([[0, 1]], dtype=int)

        with pytest.raises(ValueError):
            normal_hash_batch_multi_seed_fast(seeds, indices, num_metrics=0)

        with pytest.raises(ValueError):
            normal_hash_batch_multi_seed_fast(seeds, indices, num_metrics=-1)


class TestWeightedStatsContract:
    """API contract tests for WeightedStats dataclass."""

    def test_weighted_stats_exists(self):
        """WeightedStats dataclass exists."""
        from enn.enn.weighted_stats import WeightedStats

        assert inspect.isclass(WeightedStats)

    def test_weighted_stats_fields(self):
        """WeightedStats has expected fields."""
        from enn.enn.weighted_stats import WeightedStats

        # Create instance with dummy data
        ws = WeightedStats(
            w_normalized=np.array([0.5, 0.5]),
            l2=np.array([1.0, 2.0]),
            mu=np.array([0.0, 1.0]),
            se=np.array([0.1, 0.2]),
            se_epi=np.array([0.1, 0.2]),
            se_ale=np.array([0.0, 0.0]),
        )

        assert hasattr(ws, "w_normalized")
        assert hasattr(ws, "l2")
        assert hasattr(ws, "mu")
        assert hasattr(ws, "se")
        assert hasattr(ws, "se_epi")
        assert hasattr(ws, "se_ale")

    def test_weighted_stats_is_frozen(self):
        """WeightedStats is frozen (immutable)."""
        from enn.enn.weighted_stats import WeightedStats

        ws = WeightedStats(
            w_normalized=np.array([0.5]),
            l2=np.array([1.0]),
            mu=np.array([0.0]),
            se=np.array([0.1]),
            se_epi=np.array([0.1]),
            se_ale=np.array([0.0]),
        )

        # Attempting to modify should raise
        with pytest.raises((AttributeError, TypeError)):
            ws.mu = np.array([1.0])


class TestEnnUtilContract:
    """API contract tests for enn_util functions."""

    def test_standardize_y_exists(self):
        """standardize_y function exists."""
        from enn.enn.enn_util import standardize_y

        assert callable(standardize_y)

    def test_standardize_y_returns_tuple(self):
        """standardize_y returns (center, scale) tuple."""
        from enn.enn.enn_util import standardize_y

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        center, scale = standardize_y(y)

        assert isinstance(center, float)
        assert isinstance(scale, float)

    def test_calculate_sobol_indices_exists(self):
        """calculate_sobol_indices function exists."""
        from enn.enn.enn_util import calculate_sobol_indices

        assert callable(calculate_sobol_indices)

    def test_calculate_sobol_indices_signature(self):
        """calculate_sobol_indices has signature (x, y) -> array."""
        from enn.enn.enn_util import calculate_sobol_indices

        sig = inspect.signature(calculate_sobol_indices)
        params = list(sig.parameters.keys())
        assert params == ["x", "y"]

    def test_pareto_front_2d_maximize_exists(self):
        """pareto_front_2d_maximize function exists."""
        from enn.enn.enn_util import pareto_front_2d_maximize

        assert callable(pareto_front_2d_maximize)

    def test_arms_from_pareto_fronts_exists(self):
        """arms_from_pareto_fronts function exists."""
        from enn.enn.enn_util import arms_from_pareto_fronts

        assert callable(arms_from_pareto_fronts)
