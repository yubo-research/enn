"""Golden fixture regression tests.

Persist representative input/output and assert current implementation
produces same outputs for same inputs. Use ENV REGENERATE_GOLDEN=1 to update fixtures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from enn.turbo.config import (
    AcqType,
    ENNFitConfig,
    ENNSurrogateConfig,
    turbo_enn_config,
)

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
REGENERATE = os.environ.get("REGENERATE_GOLDEN") == "1"


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def _run_optimizer_and_capture(
    bounds, config, seed: int, num_cycles: int, num_arms: int
):
    """Run optimizer, return captured ask outputs and tr_lengths."""
    from .optimizer_parity_helpers import get_rust_optimizer

    opt = get_rust_optimizer(bounds, config, seed)
    np.random.default_rng(seed)
    asks = []
    tr_lengths = []
    for _ in range(num_cycles):
        x = opt.ask(num_arms=num_arms)
        y = _obj(x)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        opt.tell(x, y)
        asks.append(x.tolist())
        tr_lengths.append(float(opt.tr_length))
    return {"asks": asks, "tr_lengths": tr_lengths}


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
def test_golden_optimizer_enn_regression():
    """Regress against golden TuRBO-ENN fixture."""
    fixture_path = FIXTURES_DIR / "golden_optimizer_enn_seed42.json"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=8,
    )
    num_cycles = 5
    num_arms = 4
    seed = 42

    captured = _run_optimizer_and_capture(bounds, config, seed, num_cycles, num_arms)

    if REGENERATE:
        baseline = {
            "description": "Golden fixture: TuRBO-ENN, seed=42, sphere objective",
            "bounds": bounds.tolist(),
            "num_init": 8,
            "num_arms": num_arms,
            "seed": seed,
            "num_cycles": num_cycles,
            **captured,
        }
        with open(fixture_path, "w") as f:
            json.dump(baseline, f, indent=2)
        pytest.skip("Regenerated golden fixture")
        return

    if not fixture_path.exists():
        pytest.skip("Golden fixture not found; run with REGENERATE_GOLDEN=1")

    with open(fixture_path) as f:
        baseline = json.load(f)

    np.testing.assert_allclose(
        np.array(captured["asks"]),
        np.array(baseline["asks"]),
        rtol=1e-14,
        atol=1e-14,
        err_msg="ask outputs differ from golden",
    )
    np.testing.assert_allclose(
        captured["tr_lengths"],
        baseline["tr_lengths"],
        rtol=1e-12,
        atol=1e-12,
        err_msg="tr_lengths differ from golden",
    )
