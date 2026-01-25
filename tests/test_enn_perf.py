from __future__ import annotations
import time
import numpy as np
import pytest
from enn import EpistemicNearestNeighbors, enn_fit
from enn.enn.enn_params import PosteriorFlags


def make_enn_demo_data(num_samples: int, k: int, noise: float, m: int = 1):
    x = np.sort(np.random.rand(num_samples + 4))
    x[-3] = x[-4]
    x[-2] = x[-4]
    x[-1] = x[-4]
    x[1] = x[0] + 0.03
    eps = np.random.randn(num_samples + 4)
    y = np.sin(2 * m * np.pi * x) + noise * eps
    yvar = (noise**2) * np.ones_like(y)
    model = EpistemicNearestNeighbors(x[:, None], y[:, None], yvar[:, None])
    rng = np.random.default_rng(0)
    params = enn_fit(
        model,
        k=k,
        num_fit_candidates=100,
        num_fit_samples=min(10, num_samples),
        rng=rng,
    )
    return x, y, model, params


def plot_enn_posterior_logic(model, params):
    x_hat = np.linspace(0.0, 1.0, 30)
    x_hat_2d = x_hat[:, None]
    posterior = model.posterior(
        x_hat_2d, params=params, flags=PosteriorFlags(exclude_nearest=False)
    )
    return posterior.mu[:, 0], posterior.se[:, 0]


@pytest.mark.skip(reason="Timing unreliable across machines; run manually to verify")
def test_enn_demo_performance():
    np.random.seed(1)
    t0 = time.time()
    x, y, model, params = make_enn_demo_data(num_samples=1_000_000, k=5, noise=0.3, m=3)
    mu, se = plot_enn_posterior_logic(model, params)
    elapsed = time.time() - t0
    print(f"\nTime taken: {elapsed:.3f} seconds")
    assert elapsed < 0.3, f"Expected < 0.3s, got {elapsed:.3f}s"
