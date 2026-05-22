from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from enn import create_optimizer
from enn.turbo.config.optimizer_config import OptimizerConfig

from .optimizer_fixture_catalog import (
    EXPECTED_OPTIMIZER_FIXTURE_NAMES,
    FIXTURE_OBJECTIVES,
    PREFIX_CONFIG,
)

PYTHON_OPTIMIZER_FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent / "fixtures" / "python_optimizer"
)
MORBO_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "morbo"

EXACT_RTOL = 1e-14
EXACT_ATOL = 1e-14
TR_RTOL = 1e-9
TR_ATOL = 1e-9


def _config_for_fixture(name: str) -> OptimizerConfig:
    prefix = fixture_name_prefix(name)
    try:
        return PREFIX_CONFIG[prefix]
    except KeyError as exc:
        raise ValueError(f"unknown fixture name {name!r}") from exc


def _fixture_dir_for_name(name: str) -> Path:
    prefix = fixture_name_prefix(name)
    if prefix.startswith("morbo_"):
        return MORBO_FIXTURES_DIR
    return PYTHON_OPTIMIZER_FIXTURES_DIR


def load_fixture(name: str) -> dict[str, Any]:
    path = _fixture_dir_for_name(name) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing optimizer replay fixture: {path}")
    with open(path) as f:
        return json.load(f)


def assert_fixture_json_invariants(data: dict[str, Any]) -> None:
    bounds = np.array(data["bounds"], dtype=float)
    objective_fn = FIXTURE_OBJECTIVES[str(data["objective"])]
    prev_tr_obs = 0
    for step in data["steps"]:
        x = np.array(step["ask"], dtype=float)
        y = np.array(step["tell_y"], dtype=float)
        assert x.shape[1] == bounds.shape[0]
        assert y.shape[0] == x.shape[0]
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))
        assert np.all(x >= bounds[:, 0] - 1e-9)
        assert np.all(x <= bounds[:, 1] + 1e-9)
        np.testing.assert_allclose(objective_fn(x), y, rtol=EXACT_RTOL, atol=EXACT_ATOL)
        assert 0.0 < step["tr_length"] <= 2.5
        tr_obs = int(step["tr_obs_count"])
        assert tr_obs >= prev_tr_obs
        prev_tr_obs = tr_obs


def assert_fixture_contracts(data: dict[str, Any], config: OptimizerConfig) -> None:
    assert_fixture_json_invariants(data)
    bounds = np.array(data["bounds"], dtype=float)
    rng = np.random.default_rng(int(data["seed"]))
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    num_arms = int(data["num_arms"])
    for step in data["steps"]:
        x_golden = np.array(step["ask"], dtype=float)
        y_golden = np.array(step["tell_y"], dtype=float)
        x = opt.ask(num_arms=num_arms)
        assert isinstance(x, np.ndarray)
        assert x.shape == (num_arms, bounds.shape[0])
        assert np.all(np.isfinite(x))
        assert np.all(x >= bounds[:, 0] - 1e-9)
        assert np.all(x <= bounds[:, 1] + 1e-9)
        np.testing.assert_allclose(x, x_golden, rtol=EXACT_RTOL, atol=EXACT_ATOL)
        opt.tell(x_golden, y_golden)
        assert int(opt.tr_obs_count) == int(step["tr_obs_count"])
        np.testing.assert_allclose(
            opt.tr_length,
            float(step["tr_length"]),
            rtol=TR_RTOL,
            atol=TR_ATOL,
        )


def list_python_optimizer_fixture_names() -> list[str]:
    return [
        name
        for name in EXPECTED_OPTIMIZER_FIXTURE_NAMES
        if not fixture_name_prefix(name).startswith("morbo_")
    ]


def list_morbo_fixture_names() -> list[str]:
    return [
        name
        for name in EXPECTED_OPTIMIZER_FIXTURE_NAMES
        if fixture_name_prefix(name).startswith("morbo_")
    ]


def list_fixture_names() -> list[str]:
    return list(EXPECTED_OPTIMIZER_FIXTURE_NAMES)


def fixture_name_prefix(name: str) -> str:
    m = re.match(r"^(.*)_seed\d+$", name)
    return m.group(1) if m else name
