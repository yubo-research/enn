from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "python_optimizer"


@pytest.mark.parametrize(
    "name",
    [
        "turbo_enn_ucb_single_seed0",
        "turbo_enn_thompson_single_seed1",
        "turbo_enn_pareto_multi_seed2",
        "turbo_zero_seed0",
        "turbo_enn_noise_aware_seed2",
    ],
)
def test_python_optimizer_fixture_invariants(name: str):
    path = FIXTURES_DIR / f"{name}.json"
    assert path.exists(), f"missing fixture {path}"
    with open(path) as f:
        data = json.load(f)
    bounds = np.array(data["bounds"], dtype=float)
    for step in data["steps"]:
        x = np.array(step["ask"], dtype=float)
        assert x.shape[1] == bounds.shape[0]
        assert np.all(np.isfinite(x))
        assert np.all(x >= bounds[:, 0] - 1e-9)
        assert np.all(x <= bounds[:, 1] + 1e-9)
        assert 0.0 < step["tr_length"] <= 2.5
        assert step["tr_obs_count"] >= 0
