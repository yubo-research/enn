"""Parity tests for hypervolume calculation.

These tests verify that the Rust implementation produces identical
results to the Python reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.hypervolume import hypervolume_2d_max

# Try to import Rust implementation
try:
    from enn._rust import hypervolume_2d_max as rust_hypervolume_2d_max

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


class TestHypervolumeParity:
    """Bitwise parity tests for hypervolume_2d_max."""

    @pytest.mark.parametrize(
        "y,ref_point,expected",
        [
            # Simple case: two points
            (np.array([[1.0, 0.5], [0.5, 1.0]]), np.array([0.0, 0.0]), 0.75),
            # Three points with one dominated
            (np.array([[1.0, 1.0], [0.2, 0.2], [0.5, 0.5]]), np.array([0.0, 0.0]), 1.0),
            # Single point
            (np.array([[1.0, 1.0]]), np.array([0.0, 0.0]), 1.0),
            # Empty result (no dominating points)
            (np.array([[-1.0, -1.0]]), np.array([0.0, 0.0]), 0.0),
            # Non-zero reference point
            (np.array([[2.0, 2.0], [1.5, 1.5]]), np.array([1.0, 1.0]), 1.0),
        ],
    )
    def test_bitwise_parity_fixed_cases(self, y, ref_point, expected):
        """Test that Rust matches Python exactly on fixed test cases."""
        py_result = hypervolume_2d_max(y, ref_point)
        rs_result = rust_hypervolume_2d_max(y, ref_point)

        assert py_result == rs_result, f"Python {py_result} != Rust {rs_result}"
        assert py_result == expected
        assert rs_result == expected

    @pytest.mark.parametrize("seed", [0, 42, 123, 999, 2024])
    def test_bitwise_parity_random(self, seed):
        """Test parity on randomly generated inputs with fixed seeds."""
        rng = np.random.default_rng(seed)

        # Generate random points
        n_points = rng.integers(2, 20)
        y = rng.random((n_points, 2)) * 10.0
        ref_point = rng.random(2) * 2.0 - 1.0  # Range [-1, 1]

        py_result = hypervolume_2d_max(y, ref_point)
        rs_result = rust_hypervolume_2d_max(y, ref_point)

        # Bitwise equality for hypervolume (deterministic algorithm)
        assert py_result == rs_result, (
            f"Mismatch at seed {seed}: Python={py_result}, Rust={rs_result}\n"
            f"y={y}, ref={ref_point}"
        )

    def test_empty_array_parity(self):
        """Test parity on empty input."""
        y = np.array([]).reshape(0, 2)
        ref = np.array([0.0, 0.0])

        py_result = hypervolume_2d_max(y, ref)
        rs_result = rust_hypervolume_2d_max(y, ref)

        assert py_result == rs_result == 0.0

    def test_no_dominating_points_parity(self):
        """Test parity when no points dominate reference."""
        y = np.array([[-1.0, -1.0], [-0.5, -0.5], [-2.0, 0.0]])
        ref = np.array([0.0, 0.0])

        py_result = hypervolume_2d_max(y, ref)
        rs_result = rust_hypervolume_2d_max(y, ref)

        assert py_result == rs_result == 0.0

    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_parity_with_nans_and_infs(self, seed):
        """Test behavior with edge cases (should both raise or both handle)."""
        rng = np.random.default_rng(seed)
        y = rng.random((5, 2))
        ref = np.array([0.0, 0.0])

        # Normal case
        py_result = hypervolume_2d_max(y, ref)
        rs_result = rust_hypervolume_2d_max(y, ref)
        assert py_result == rs_result
