#!/usr/bin/env python

from __future__ import annotations

import tempfile
import time

import click
import numpy as np

from enn import create_optimizer, turbo_enn_config
from enn.benchmarks import Ackley
from enn.turbo.config import (
    AcqType,
    CandidateGenConfig,
    ENNFitConfig,
    ENNIndexDriver,
    ENNSurrogateConfig,
    TurboTRConfig,
)
from enn.turbo.optimizer_config import CandidateRV

NUM_DIM = 99
NUM_ROUNDS = 101
NUM_ARMS = 100
ACKLEY_NOISE = 0.1
SEED = 0
INDEX_TYPE_CHOICES: tuple[str, ...] = ("flat", "bpann_disk")


def parse_index_driver(name: str) -> ENNIndexDriver:
    mapping = {
        "flat": ENNIndexDriver.FLAT,
        "bpann_disk": ENNIndexDriver.BPANN_DISK,
    }
    if name not in mapping:
        raise ValueError(f"Unknown index type: {name}")
    return mapping[name]


def build_ackley_enn_surrogate(index_driver: ENNIndexDriver) -> ENNSurrogateConfig:
    """Build ENN surrogate config; BPANN_DISK requires disk storage + work_dir."""
    enn_kwargs: dict[str, object] = {
        "k": 10,
        "fit": ENNFitConfig(num_fit_samples=100),
        "index_driver": index_driver,
    }
    if index_driver == ENNIndexDriver.BPANN_DISK:
        enn_kwargs["enn_storage"] = "disk"
        enn_kwargs["work_dir"] = tempfile.mkdtemp(prefix="qa_ackley_bpann_")
    return ENNSurrogateConfig(**enn_kwargs)


def run_turbo_enn_ackley(
    *,
    index_driver: ENNIndexDriver,
    num_dim: int = NUM_DIM,
    num_rounds: int = NUM_ROUNDS,
    num_arms: int = NUM_ARMS,
    noise: float = ACKLEY_NOISE,
    seed: int = SEED,
) -> None:
    """Run TuRBO-ENN on Ackley; print EVAL metrics each round."""
    rng = np.random.default_rng(seed)
    objective = Ackley(noise=noise, rng=rng)
    bounds = np.array([objective.bounds] * num_dim, dtype=float)
    config = turbo_enn_config(
        enn=build_ackley_enn_surrogate(index_driver),
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
    cumulative_num_arms = 0
    t_start = time.perf_counter()
    for i_iter in range(num_rounds):
        x_arms = optimizer.ask(num_arms=num_arms)
        y_obs = np.asarray(objective(x_arms), dtype=float).reshape(-1, 1)
        optimizer.tell(x_arms, y_obs, y_var=y_var_scale * np.ones_like(y_obs))
        y_best = max(y_best, float(np.max(y_obs)))
        cumulative_num_arms += int(np.asarray(x_arms).shape[0])
        seconds_since_start_of_run = time.perf_counter() - t_start
        click.echo(
            f"EVAL: iter = {i_iter} arms = {cumulative_num_arms} "
            f"dt = {seconds_since_start_of_run:.02f} y_best = {y_best:.04f}"
        )


@click.group()
def cli() -> None:
    """QA / optimization-performance checks."""


@cli.command("ackley-99")
@click.argument("index_type", type=click.Choice(INDEX_TYPE_CHOICES))
def ackley_99(index_type: str) -> None:
    """TuRBO-ENN optimization-performance check on 99-D Ackley."""
    run_turbo_enn_ackley(index_driver=parse_index_driver(index_type))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
