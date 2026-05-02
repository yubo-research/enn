"""Parity tests for utility functions.

These tests verify that the Rust implementation produces identical
results to the Python reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_util import (
    calculate_sobol_indices,
    pareto_front_2d_maximize,
    standardize_y,
)

# Try to import Rust implementation
try:
    from enn._rust import (
        calculate_sobol_indices as rust_calculate_sobol_indices,
    )
    from enn._rust import (
        pareto_front_2d_maximize as rust_pareto_front,
    )
    from enn._rust import (
        standardize_y as rust_standardize_y,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


class TestStandardizeYParity:
    """Parity tests for standardize_y function."""

    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_standardize_y_parity(self, seed):
        """Test standardize_y produces same center and scale."""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(100)

        py_center, py_scale = standardize_y(y)
        rs_center, rs_scale = rust_standardize_y(y)

        # Bitwise equality for these deterministic calculations
        assert py_center == rs_center, f"Center mismatch: {py_center} vs {rs_center}"
        np.testing.assert_allclose(
            py_scale,
            rs_scale,
            rtol=1e-14,
            atol=0.0,
            err_msg=f"Scale mismatch: {py_scale} vs {rs_scale}",
        )

    def test_standardize_y_constant_parity(self):
        """Test with constant input."""
        y = np.array([5.0, 5.0, 5.0, 5.0])

        py_center, py_scale = standardize_y(y)
        rs_center, rs_scale = rust_standardize_y(y)

        assert py_center == rs_center
        assert py_scale == rs_scale == 1.0

    def test_standardize_y_empty_parity(self):
        """Test with empty input."""
        y = np.array([])

        py_center, py_scale = standardize_y(y)
        rs_center, rs_scale = rust_standardize_y(y)

        assert np.isnan(py_center)
        assert np.isnan(rs_center)
        assert py_scale == rs_scale == 1.0


class TestParetoFrontParity:
    """Parity tests for pareto_front_2d_maximize."""

    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_pareto_front_parity_random(self, seed):
        """Test Pareto front on random data."""
        rng = np.random.default_rng(seed)

        n = 20
        a = rng.random(n) * 10.0
        b = rng.random(n) * 10.0

        py_front = pareto_front_2d_maximize(a, b)
        rs_front = rust_pareto_front(a, b)

        # Should produce same indices
        np.testing.assert_array_equal(py_front, rs_front)

    def test_pareto_front_known_cases(self):
        """Test on known cases."""
        # Case 1: Simple front
        a = np.array([1.0, 0.5, 0.2])
        b = np.array([0.5, 1.0, 0.2])

        py_front = pareto_front_2d_maximize(a, b)
        rs_front = rust_pareto_front(a, b)

        np.testing.assert_array_equal(py_front, rs_front)
        assert set(py_front) == {0, 1}  # Points 0 and 1 are on front

    def test_pareto_front_empty_parity(self):
        """Test with empty arrays."""
        a = np.array([])
        b = np.array([])

        py_front = pareto_front_2d_maximize(a, b)
        rs_front = rust_pareto_front(a, b)

        assert len(py_front) == len(rs_front) == 0


class TestSobolIndicesParity:
    """Parity tests for calculate_sobol_indices.

    Note: Sobol indices involve floating point computations that may
    have small differences between Python and Rust implementations.
    We use tolerance-based checks rather than bitwise equality.
    """

    def test_sobol_indices_parity(self):
        """Test Sobol indices with tolerance."""
        rng = np.random.default_rng(42)
        x = rng.random((80, 3))
        y = x[:, 0] ** 2 + 0.3 * x[:, 1] + 0.01 * rng.standard_normal(80)

        py_result = calculate_sobol_indices(x, y)
        rs_result = rust_calculate_sobol_indices(x, y)

        np.testing.assert_allclose(py_result, rs_result, rtol=1e-6, atol=1e-8)
