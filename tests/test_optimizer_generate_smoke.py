from __future__ import annotations

import numpy as np

from enn.turbo.config import turbo_zero_config
from enn.turbo.python_fallback.optimizer_generate import (
    _CandidateGenContext,
    generate_optimizer_candidates,
)
from enn.turbo.python_fallback.turbo_trust_region import TurboTrustRegion


def test_generate_optimizer_candidates_sobol_path():
    cfg = turbo_zero_config(num_init=2)
    np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    tr = TurboTrustRegion(config=cfg.trust_region, num_dim=2)
    rng = np.random.default_rng(0)
    ctx = _CandidateGenContext(
        config=cfg,
        tr_state=tr,
        num_dim=2,
        sobol_seed_base=1,
        restart_generation=0,
        rng=rng,
    )
    x0 = np.array([0.5, 0.5])
    out = generate_optimizer_candidates(ctx, x0, None, n_obs=0, num_arms=2)
    assert out.ndim == 2 and out.shape[1] == 2
