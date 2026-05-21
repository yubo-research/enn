from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.config import AcqType, ENNFitConfig, ENNSurrogateConfig, turbo_enn_config

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")

EXACT_RTOL = 1e-14
EXACT_ATOL = 1e-14


def test_rust_backend_local_determinism():
    from .optimizer_parity_helpers import get_rust_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=6,
    )
    opt_a = get_rust_optimizer(bounds, config, seed=99)
    opt_b = get_rust_optimizer(bounds, config, seed=99)
    xa = opt_a.ask(num_arms=3)
    xb = opt_b.ask(num_arms=3)
    np.testing.assert_allclose(xa, xb, rtol=EXACT_RTOL, atol=EXACT_ATOL)
