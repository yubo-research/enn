from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from enn import create_optimizer
from enn.turbo.config import (
    AcqType,
    ENNFitConfig,
    ENNSurrogateConfig,
    turbo_enn_config,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimizer_quality_common import run_best_y  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def assert_ci_quality_gate() -> None:
    path = FIXTURES_DIR / "optimizer_quality_baseline.json"
    with open(path) as f:
        baseline = json.load(f)
    y_range = float(baseline["y_range"])
    tol = 0.02 * y_range
    failures = []
    for cell in baseline["cells"]:
        dim = int(cell["dim"])
        seed = int(cell["seed"])
        budget = int(cell["budget"])
        py_best = float(cell["python_best_y"])
        bounds = np.tile(np.array([[0.0, 1.0]], dtype=float), (dim, 1))
        config = turbo_enn_config(
            acq_type=AcqType.UCB,
            enn=ENNSurrogateConfig(k=10, fit=ENNFitConfig(num_fit_samples=10)),
            num_init=min(10, 2 * dim),
        )
        rust_best = run_best_y(
            bounds,
            config,
            seed,
            budget=budget,
            num_arms=2,
            create_optimizer=create_optimizer,
        )
        if rust_best < py_best - tol:
            failures.append((dim, seed, py_best, rust_best))
    if failures:
        msg = "; ".join(
            f"d={d} seed={s} py={py:.4f} rust={r:.4f}" for d, s, py, r in failures[:5]
        )
        raise AssertionError(f"quality gate failures: {msg}")
