"""Parity tests for hash-based RNG functions.

These tests verify that the Rust implementation produces identical
outputs to the Python reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_hash import (
    normal_hash_batch_multi_seed_fast,
)

# Try to import Rust implementation
try:
    from enn._rust import normal_hash_batch_multi_seed_fast as rust_hash_fast

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


class TestHashParity:
    """Bitwise parity tests for hash-based RNG."""

    @pytest.mark.parametrize("seed", [0, 42, 123, 999, 2024])
    def test_fast_hash_bitwise_parity_simple(self, seed):
        """Test fast hash produces identical outputs."""
        function_seeds = np.array([seed], dtype=np.int64)
        data_indices = np.array([[0, 1, 2]], dtype=int)
        num_metrics = 3

        py_result = normal_hash_batch_multi_seed_fast(
            function_seeds, data_indices, num_metrics
        )
        rs_result = rust_hash_fast(function_seeds, data_indices, num_metrics)

        # Must be exact match
        np.testing.assert_array_equal(py_result, rs_result)

    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_fast_hash_multiple_seeds(self, seed):
        """Test with multiple function seeds."""
        rng = np.random.default_rng(seed)

        num_seeds = rng.integers(1, 5)
        function_seeds = rng.integers(0, 1000, size=num_seeds).astype(np.int64)
        data_indices = rng.integers(0, 100, size=(3, 4)).astype(int)
        num_metrics = rng.integers(1, 5)

        py_result = normal_hash_batch_multi_seed_fast(
            function_seeds, data_indices, num_metrics
        )
        rs_result = rust_hash_fast(function_seeds, data_indices, num_metrics)

        np.testing.assert_array_equal(py_result, rs_result)
        assert py_result.shape == (num_seeds, 3, 4, num_metrics)

    @pytest.mark.parametrize("seed", [0, 42])
    def test_fast_hash_large_arrays(self, seed):
        """Test with larger arrays."""
        rng = np.random.default_rng(seed)

        function_seeds = np.array([42, 99, 123], dtype=np.int64)
        data_indices = rng.integers(0, 50, size=(5, 6)).astype(int)
        num_metrics = 4

        py_result = normal_hash_batch_multi_seed_fast(
            function_seeds, data_indices, num_metrics
        )
        rs_result = rust_hash_fast(function_seeds, data_indices, num_metrics)

        np.testing.assert_array_equal(py_result, rs_result)

    @pytest.mark.parametrize("seed", [0, 42])
    def test_determinism_parity(self, seed):
        """Both implementations should be deterministic."""
        function_seeds = np.array([seed], dtype=np.int64)
        data_indices = np.array([[0, 1]], dtype=int)
        num_metrics = 2

        # Multiple calls should give same result
        py_1 = normal_hash_batch_multi_seed_fast(
            function_seeds, data_indices, num_metrics
        )
        py_2 = normal_hash_batch_multi_seed_fast(
            function_seeds, data_indices, num_metrics
        )
        rs_1 = rust_hash_fast(function_seeds, data_indices, num_metrics)
        rs_2 = rust_hash_fast(function_seeds, data_indices, num_metrics)

        np.testing.assert_array_equal(py_1, py_2)
        np.testing.assert_array_equal(rs_1, rs_2)
        np.testing.assert_array_equal(py_1, rs_1)

    def test_duplicate_indices_parity(self):
        """Test handling of duplicate data indices."""
        function_seeds = np.array([42], dtype=np.int64)
        data_indices = np.array([[0, 0, 1, 1, 2]], dtype=int)  # Duplicates
        num_metrics = 2

        py_result = normal_hash_batch_multi_seed_fast(
            function_seeds, data_indices, num_metrics
        )
        rs_result = rust_hash_fast(function_seeds, data_indices, num_metrics)

        np.testing.assert_array_equal(py_result, rs_result)

        # Verify duplicates produce same values
        assert np.allclose(py_result[0, 0, 0, :], py_result[0, 0, 1, :])
        assert np.allclose(rs_result[0, 0, 0, :], rs_result[0, 0, 1, :])

    def test_statistical_properties_parity(self):
        """Verify both produce standard normal (mean ~0, std ~1)."""
        function_seeds = np.array([0], dtype=np.int64)
        data_indices = np.arange(1000).reshape(10, 100).astype(int)
        num_metrics = 1

        py_result = normal_hash_batch_multi_seed_fast(
            function_seeds, data_indices, num_metrics
        )
        rs_result = rust_hash_fast(function_seeds, data_indices, num_metrics)

        # Both should have similar statistical properties
        py_mean = np.mean(py_result)
        py_std = np.std(py_result)
        rs_mean = np.mean(rs_result)
        rs_std = np.std(rs_result)

        assert abs(py_mean) < 0.1, f"Python mean {py_mean} too far from 0"
        assert abs(rs_mean) < 0.1, f"Rust mean {rs_mean} too far from 0"
        assert 0.9 < py_std < 1.1, f"Python std {py_std} too far from 1"
        assert 0.9 < rs_std < 1.1, f"Rust std {rs_std} too far from 1"


class TestHashSlowVsFastParity:
    """Verify slow (Philox) and fast (SplitMix) versions match where applicable."""

    @pytest.mark.skip(
        reason="Philox and SplitMix64 produce different sequences - design difference"
    )
    def test_slow_vs_fast_comparison(self):
        """Note: Slow and fast versions use different RNGs, so outputs differ."""
        pass
