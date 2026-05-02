from __future__ import annotations

import numpy as np

from enn.turbo.rust_optimizer import (
    RustOptimizer,
    _ObsView,
    create_optimizer,
)


def test_rust_optimizer_kiss_surface_has_view_and_factory():
    from enn.turbo import rust_optimizer as ro

    assert RustOptimizer.__init__ is not None
    assert create_optimizer is ro.create_optimizer
    v = _ObsView(np.array([[0.0]]))
    assert v.view().shape == (1, 1)
    bounds = np.array([[0.0, 1.0]], dtype=float)
    cfg = __import__(
        "enn.turbo.config", fromlist=["turbo_zero_config"]
    ).turbo_zero_config(num_init=1)
    create_optimizer(bounds=bounds, config=cfg, rng=np.random.default_rng(0))
