"""Closed-loop maximization on known-optimum objectives at a fixed evaluation budget."""

import time

import numpy as np

from enn import create_optimizer, turbo_enn_config
from enn.evals.metrics import format_smaller
from enn.turbo.config import AcqType, ENNFitConfig, ENNSurrogateConfig
from enn.turbo.config.candidate_gen_config import CandidateGenConfig
from ops.stress import format_mean_se, mean_se

NUM_DIM = 5
BUDGET = 48
NUM_INIT = 10
SEEDS = (0, 1, 2)
CHECKPOINTS = (NUM_INIT, 24, BUDGET)


def objective(x: np.ndarray, problem: str) -> np.ndarray:
    """Negative losses with maximum zero at the origin, on [-1, 1]^5."""
    if problem == "sphere":
        loss = np.mean(x**2, axis=1)
    elif problem == "ackley":
        z = 5 * x
        loss = (
            -20 * np.exp(-0.2 * np.sqrt(np.mean(z**2, axis=1)))
            - np.exp(np.mean(np.cos(2 * np.pi * z), axis=1))
            + 20
            + np.e
        )
    else:
        raise ValueError(f"unknown problem: {problem}")
    return -loss.reshape(-1, 1)


def build_optimizer(seed: int):
    config = turbo_enn_config(
        num_init=NUM_INIT,
        acq_type=AcqType.UCB,
        candidates=CandidateGenConfig(max_candidates=128),
        enn=ENNSurrogateConfig(
            k=8, fit=ENNFitConfig(num_fit_samples=24, num_fit_candidates=16)
        ),
    )
    return create_optimizer(
        bounds=np.array([[-1.0, 1.0]] * NUM_DIM),
        config=config,
        rng=np.random.default_rng(seed),
    )


def run_seed(problem: str, seed: int) -> list[dict]:
    """Share the initial design; spend every subsequent evaluation on an actual proposal."""
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    optimizer = build_optimizer(seed)
    initial = rng.uniform(-1, 1, size=(NUM_INIT, NUM_DIM))
    y = objective(initial, problem)
    optimizer.tell(initial, y)
    best = float(y.max())
    rows = [
        {
            "method": "enn",
            "n": NUM_INIT,
            "regret": max(0.0, -best),
            "wall_s": time.perf_counter() - start,
        }
    ]
    for n in range(NUM_INIT + 1, BUDGET + 1):
        x = optimizer.ask(num_arms=1)
        y = objective(x, problem)
        optimizer.tell(x, y)
        best = max(best, float(y.max()))
        if n in CHECKPOINTS:
            rows.append(
                {
                    "method": "enn",
                    "n": n,
                    "regret": max(0.0, -best),
                    "wall_s": time.perf_counter() - start,
                }
            )
    return rows + random_baseline(problem, seed, initial)


def random_baseline(problem: str, seed: int, initial: np.ndarray) -> list[dict]:
    start = time.perf_counter()
    rng = np.random.default_rng(seed + 1000)
    best = float(objective(initial, problem).max())
    rows = []
    previous = NUM_INIT
    for n in CHECKPOINTS:
        if n > previous:
            x = rng.uniform(-1, 1, size=(n - previous, NUM_DIM))
            best = max(best, float(objective(x, problem).max()))
        rows.append(
            {
                "method": "random",
                "n": n,
                "regret": max(0.0, -best),
                "wall_s": time.perf_counter() - start,
            }
        )
        previous = n
    return rows


def run_optimization() -> None:
    print(f"dimensions={NUM_DIM} budget={BUDGET} num_init={NUM_INIT} seeds={SEEDS}")
    for problem in ("sphere", "ackley"):
        rows = [row for seed in SEEDS for row in run_seed(problem, seed)]
        for method in ("enn", "random"):
            for n in CHECKPOINTS:
                selected = [r for r in rows if r["method"] == method and r["n"] == n]
                metrics = " ".join(
                    format_smaller(
                        key, format_mean_se(mean_se([r[key] for r in selected]))
                    )
                    for key in ("regret", "wall_s")
                )
                print(
                    f"EVAL: problem = {problem} method = {method} n = {n} {metrics}",
                    flush=True,
                )
