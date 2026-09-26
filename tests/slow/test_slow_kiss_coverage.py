from __future__ import annotations

import numpy as np
import pytest


def _fit_gp_surrogate_for_kiss(rng):
    from enn.turbo.python_fallback.components.gp_surrogate import GPSurrogate

    surrogate = GPSurrogate()
    x = np.array([[0.2, 0.3], [0.5, 0.5], [0.7, 0.8]], dtype=float)
    y = np.array([0.5, 0.7, 0.3], dtype=float)
    surrogate.fit(x, y, None, num_steps=2, rng=rng)
    return surrogate


@pytest.mark.slow
def test_pareto_and_random_acq_optimizer_select():
    from enn.turbo.python_fallback.components.pareto_acq_optimizer import (
        ParetoAcqOptimizer,
    )
    from enn.turbo.python_fallback.components.random_acq_optimizer import (
        RandomAcqOptimizer,
    )

    rng = np.random.default_rng(0)
    surrogate = _fit_gp_surrogate_for_kiss(rng)
    x_cand = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=float)
    assert ParetoAcqOptimizer().select(x_cand, 2, surrogate, rng).shape == (2, 2)
    assert RandomAcqOptimizer().select(x_cand, 2, surrogate, rng).shape == (2, 2)


@pytest.mark.slow
def test_turbo_gp_base():
    from enn.turbo.python_fallback.turbo_gp_base import TurboGPBase

    assert hasattr(TurboGPBase, "forward")
