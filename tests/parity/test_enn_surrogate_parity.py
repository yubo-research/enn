"""ENNSurrogate / ENN model cross-language parity: Python vs Rust."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from enn._rust import EpistemicNearestNeighbors as RustENN

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def test_enn_posterior_rust_vs_python():
    """Rust ENN posterior matches Python ENN posterior on same data.

    Cross-language parity for the core ENN model used by ENNSurrogate.
    """
    from enn import EpistemicNearestNeighbors as PyENN
    from enn.enn.enn_params import ENNParams, PosteriorFlags

    train_x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    params = ENNParams(
        k_num_neighbors=2,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
    )
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=False)

    py_model = PyENN(train_x, train_y, scale_x=False)
    py_out = py_model.posterior(query, params=params, flags=flags)

    rust_model = RustENN(train_x, train_y, scale_x=False, index_driver="Exact")
    rs_mu, rs_se, _ = rust_model.posterior(
        query,
        k_num_neighbors=2,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
        exclude_nearest=False,
        observation_noise=False,
    )

    np.testing.assert_allclose(py_out.mu, rs_mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(py_out.se, rs_se, rtol=1e-12, atol=1e-12)
