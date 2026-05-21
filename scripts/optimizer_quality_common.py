from __future__ import annotations

from typing import Any, Callable

import numpy as np


def sphere_objective(x: np.ndarray) -> np.ndarray:
    return (-np.sum((x - 0.5) ** 2, axis=1)).reshape(-1, 1)


def run_best_y(
    bounds: np.ndarray,
    config: Any,
    seed: int,
    budget: int,
    num_arms: int,
    *,
    create_optimizer: Callable[..., Any],
) -> float:
    rng = np.random.default_rng(seed)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    evals = 0
    best = -np.inf
    while evals < budget:
        n = min(num_arms, budget - evals)
        x = opt.ask(num_arms=n)
        y = sphere_objective(x)
        opt.tell(x, y)
        evals += n
        best = max(best, float(np.max(y)))
    return best
