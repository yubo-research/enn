from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enn import create_optimizer  # noqa: E402
from enn.turbo.config import (  # noqa: E402
    AcqType,
    ENNFitConfig,
    ENNSurrogateConfig,
    turbo_enn_config,
)
import enn.turbo.rust_optimizer as ro  # noqa: E402

CI_DIMS = (2, 8)
CI_SEEDS = tuple(range(5))
FULL_DIMS = (2, 8, 32)
FULL_SEEDS = tuple(range(20))


def _sphere(x: np.ndarray) -> np.ndarray:
    return (-np.sum((x - 0.5) ** 2, axis=1)).reshape(-1, 1)


def _run_best_y(bounds, config, seed: int, budget: int, num_arms: int) -> float:
    rng = np.random.default_rng(seed)
    with patch.object(ro, "is_rust_supported_config", return_value=False):
        opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    evals = 0
    best = -np.inf
    while evals < budget:
        n = min(num_arms, budget - evals)
        x = opt.ask(num_arms=n)
        y = _sphere(x)
        opt.tell(x, y)
        evals += n
        best = max(best, float(np.max(y)))
    return best


def _build_cells(dims: tuple[int, ...], seeds: tuple[int, ...]) -> list[dict]:
    cells = []
    for dim in dims:
        bounds = np.tile(np.array([[0.0, 1.0]], dtype=float), (dim, 1))
        config = turbo_enn_config(
            acq_type=AcqType.UCB,
            enn=ENNSurrogateConfig(k=10, fit=ENNFitConfig(num_fit_samples=10)),
            num_init=min(10, 2 * dim),
        )
        for seed in seeds:
            cells.append(
                {
                    "objective": "sphere_centered_0.5",
                    "dim": dim,
                    "budget": 64,
                    "seed": seed,
                    "acquisition": "ucb",
                    "python_best_y": _run_best_y(
                        bounds, config, seed, budget=64, num_arms=2
                    ),
                }
            )
    return cells


def main() -> None:
    out = ROOT / "tests" / "fixtures" / "optimizer_quality_baseline.json"
    payload = {
        "y_range": 1.0,
        "ci": {"dims": list(CI_DIMS), "seeds": list(CI_SEEDS)},
        "cells": _build_cells(CI_DIMS, CI_SEEDS),
        "full_cells": _build_cells(FULL_DIMS, FULL_SEEDS),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", out, "cells", len(payload["cells"]))


if __name__ == "__main__":
    main()
