"""Skipping the f64 re-rank of budget-stopped queries must not change posteriors beyond float32 rounding."""

from __future__ import annotations

import numpy as np

from enn import EpistemicNearestNeighbors, _rust
from enn.enn.enn_params import ENNParams
from evals.harness.work_dirs import fresh_work_dir


def test_f32_rerank_skip_matches_exact_rerank_to_float32_precision():
    d, n = 20, 12000
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d))
    y = np.sin(x[:, :1]) + 0.1 * rng.standard_normal((n, 1))
    params = ENNParams(
        k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    model.ensure_index_sync()
    q = rng.standard_normal((40, d))
    out = {}
    try:
        for skip in (True, False):
            _rust.set_f32_rerank_skip(skip)
            out[skip] = model.posterior(q, params=params)
    finally:
        _rust.set_f32_rerank_skip(True)
    np.testing.assert_allclose(out[True].mu, out[False].mu, rtol=0, atol=1e-5)
    np.testing.assert_allclose(out[True].se, out[False].se, rtol=1e-5, atol=1e-6)
