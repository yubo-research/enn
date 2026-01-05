"""Minimal test to reproduce TURBO_ONE + MORBO issue."""

import numpy as np

from enn import create_optimizer
from enn.benchmarks import DoubleAckley
from enn.turbo.config import MorboTRConfig, turbo_one_config


def test_morbo_turbo_one_two_rounds():
    """Run TURBO_ONE with MORBO for 2 rounds."""
    num_dim = 30
    num_arms = 10
    noise = 0.1
    num_metrics = 2

    rng = np.random.default_rng(42)
    objective = DoubleAckley(noise=noise, rng=rng)
    bounds = np.array([objective.bounds] * num_dim, dtype=float)

    config = turbo_one_config(trust_region=MorboTRConfig(num_metrics=num_metrics))
    optimizer = create_optimizer(bounds=bounds, config=config, rng=rng)

    # Run enough iterations for GP to kick in (usually after ~10-20 points)
    for iteration in range(10):
        x_arms = optimizer.ask(num_arms=num_arms)
        y_obs = objective(x_arms)
        optimizer.tell(x_arms, y_obs)
        print(
            f"Iteration {iteration}: x_arms shape = {x_arms.shape}, y_obs shape = {y_obs.shape}"
        )
