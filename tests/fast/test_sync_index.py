from __future__ import annotations

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams


def test_ensure_index_sync_callable():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 2))
    y = rng.standard_normal((8, 1))
    model = EpistemicNearestNeighbors(x, y)
    model.ensure_index_sync()


def test_ensure_index_sync_after_add_matches_fresh_posterior():
    rng = np.random.default_rng(42)
    d = 3
    x0 = rng.standard_normal((10, d))
    y0 = rng.standard_normal((10, 1))
    x1 = rng.standard_normal((5, d))
    y1 = rng.standard_normal((5, 1))

    inc = EpistemicNearestNeighbors(x0, y0)
    inc.add(x1, y1)
    inc.ensure_index_sync()

    fresh = EpistemicNearestNeighbors(np.vstack([x0, x1]), np.vstack([y0, y1]))
    params = ENNParams(
        k_num_neighbors=3, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    x_test = rng.standard_normal((4, d))

    post_inc = inc.posterior(x_test, params=params)
    post_fresh = fresh.posterior(x_test, params=params)
    np.testing.assert_allclose(post_inc.mu, post_fresh.mu, rtol=1e-6)
    np.testing.assert_allclose(post_inc.se, post_fresh.se, rtol=1e-6)


def test_ensure_index_sync_twice_without_add():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((6, 2))
    y = rng.standard_normal((6, 1))
    model = EpistemicNearestNeighbors(x, y)
    model.add(rng.standard_normal((2, 2)), rng.standard_normal((2, 1)))
    model.ensure_index_sync()
    model.ensure_index_sync()
