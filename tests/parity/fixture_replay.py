from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from enn import create_optimizer
from enn.turbo.config import (
    AcqType,
    CandidateGenConfig,
    ENNFitConfig,
    ENNSurrogateConfig,
    MorboTRConfig,
    MultiObjectiveConfig,
    TurboTRConfig,
    lhd_only_config,
    turbo_enn_config,
    turbo_zero_config,
)
from enn.turbo.config.optimizer_config import OptimizerConfig

PYTHON_OPTIMIZER_FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent / "fixtures" / "python_optimizer"
)
MORBO_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "morbo"

EXACT_RTOL = 1e-14
EXACT_ATOL = 1e-14
TR_RTOL = 1e-9
TR_ATOL = 1e-9

_PREFIX_CONFIG: dict[str, OptimizerConfig] = {
    "turbo_enn_ucb_single": turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=3,
    ),
    "turbo_enn_thompson_single": turbo_enn_config(
        acq_type=AcqType.THOMPSON,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=3,
    ),
    "turbo_enn_pareto_multi": turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=2,
    ),
    "turbo_zero": turbo_zero_config(num_init=3),
    "turbo_enn_trailing_obs": turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=2,
        trailing_obs=12,
    ),
    "turbo_enn_noise_aware": turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=8)),
        trust_region=TurboTRConfig(noise_aware=True),
        num_init=2,
    ),
    "lhd_only": lhd_only_config(num_init=3),
    "morbo_enn_separable_unimodal": turbo_enn_config(
        acq_type=AcqType.THOMPSON,
        enn=ENNSurrogateConfig(
            k=4,
            fit=ENNFitConfig(num_fit_samples=4, num_fit_candidates=8),
        ),
        trust_region=MorboTRConfig(
            multi_objective=MultiObjectiveConfig(num_metrics=2),
            noise_aware=True,
        ),
        num_init=8,
        candidates=CandidateGenConfig(num_candidates=64),
    ),
}


def _config_for_fixture(name: str) -> OptimizerConfig:
    prefix = fixture_name_prefix(name)
    try:
        return _PREFIX_CONFIG[prefix]
    except KeyError as exc:
        raise ValueError(f"unknown fixture name {name!r}") from exc


def _fixture_dir_for_name(name: str) -> Path:
    prefix = fixture_name_prefix(name)
    if prefix.startswith("morbo_"):
        return MORBO_FIXTURES_DIR
    return PYTHON_OPTIMIZER_FIXTURES_DIR


def load_fixture(name: str) -> dict[str, Any]:
    path = _fixture_dir_for_name(name) / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def assert_fixture_contracts(data: dict[str, Any], config: OptimizerConfig) -> None:
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
    return sorted(p.stem for p in PYTHON_OPTIMIZER_FIXTURES_DIR.glob("*.json"))


def list_morbo_fixture_names() -> list[str]:
    return sorted(p.stem for p in MORBO_FIXTURES_DIR.glob("*.json"))


def list_fixture_names() -> list[str]:
    return list_python_optimizer_fixture_names() + list_morbo_fixture_names()


def fixture_name_prefix(name: str) -> str:
    m = re.match(r"^(.*)_seed\d+$", name)
    return m.group(1) if m else name
