from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_turbo_config = importlib.import_module("enn.turbo.config")
AcqType = _turbo_config.AcqType
ENNFitConfig = _turbo_config.ENNFitConfig
ENNSurrogateConfig = _turbo_config.ENNSurrogateConfig
turbo_enn_config = _turbo_config.turbo_enn_config
create_optimizer = importlib.import_module("enn").create_optimizer
_spec = importlib.util.spec_from_file_location(
    "optimizer_quality_common",
    Path(__file__).resolve().parent / "optimizer_quality_common.py",
)
_oqc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_oqc)
run_best_y = _oqc.run_best_y

CI_DIMS = (2, 8)
CI_SEEDS = tuple(range(5))
FULL_DIMS = (2, 8, 32)
FULL_SEEDS = tuple(range(20))


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
                    "python_best_y": run_best_y(
                        bounds,
                        config,
                        seed,
                        budget=64,
                        num_arms=2,
                        create_optimizer=create_optimizer,
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
