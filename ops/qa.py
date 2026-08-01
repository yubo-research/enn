#!/usr/bin/env python

from __future__ import annotations

import click
import numpy as np

from enn import create_optimizer, turbo_enn_config
from enn.benchmarks import Ackley
from enn.turbo.config import (
    AcqType,
    CandidateGenConfig,
    ENNFitConfig,
    ENNSurrogateConfig,
    TurboTRConfig,
)
from enn.turbo.optimizer_config import CandidateRV

NUM_DIM = 99
NUM_ROUNDS = 101
NUM_ARMS = 100
ACKLEY_NOISE = 0.1
SEED = 0


def run_turbo_enn_ackley(
    *,
    num_dim: int = NUM_DIM,
    num_rounds: int = NUM_ROUNDS,
    num_arms: int = NUM_ARMS,
    noise: float = ACKLEY_NOISE,
    seed: int = SEED,
) -> None:
    """Run TuRBO-ENN on Ackley and print best-so-far y each round."""
    rng = np.random.default_rng(seed)
    objective = Ackley(noise=noise, rng=rng)
    bounds = np.array([objective.bounds] * num_dim, dtype=float)
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(k=10, fit=ENNFitConfig(num_fit_samples=100)),
        candidates=CandidateGenConfig(candidate_rv=CandidateRV.UNIFORM),
        trust_region=TurboTRConfig(noise_aware=True),
        acq_type=AcqType.UCB,
    )
    optimizer = create_optimizer(
        bounds=bounds,
        config=config,
        rng=np.random.default_rng(seed),
    )
    y_var_scale = float(noise) ** 2
    y_best = -np.inf
    for i_iter in range(num_rounds):
        x_arms = optimizer.ask(num_arms=num_arms)
        y_obs = np.asarray(objective(x_arms), dtype=float).reshape(-1, 1)
        optimizer.tell(x_arms, y_obs, y_var=y_var_scale * np.ones_like(y_obs))
        y_best = max(y_best, float(np.max(y_obs)))
        click.echo(f"EVAL: {i_iter:d} {y_best:.04f}")


@click.command()
def main() -> None:
    """TuRBO-ENN optimization-performance check on 99-D Ackley."""
    run_turbo_enn_ackley()


if __name__ == "__main__":
    main()
