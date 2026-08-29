from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from enn.benchmarks.separable_unimodal import separable_unimodal_objective
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

UNIT_BOX_BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
MORBO_SEPARABLE_BOUNDS = np.array([[-300.0, 600.0], [0.2, 1.0]], dtype=float)


def sphere_centered_objective(x: np.ndarray) -> np.ndarray:
    return (-np.sum((x - 0.5) ** 2, axis=1)).reshape(-1, 1)


FIXTURE_OBJECTIVES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sphere_centered_0.5": sphere_centered_objective,
    "separable_unimodal_two_metric": separable_unimodal_objective,
}


@dataclass(frozen=True)
class FixtureRunSpec:
    num_cycles: int
    num_arms: int
    objective: str


@dataclass(frozen=True)
class FixtureGeneratorEntry:
    prefix: str
    config_key: str
    bounds: np.ndarray
    run: FixtureRunSpec
    morbo: bool = False


PREFIX_CONFIG: dict[str, OptimizerConfig] = {
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


FIXTURE_GENERATOR_ENTRIES: tuple[FixtureGeneratorEntry, ...] = (
    FixtureGeneratorEntry(
        "turbo_enn_ucb_single_seed",
        "turbo_enn_ucb_single",
        UNIT_BOX_BOUNDS,
        FixtureRunSpec(4, 3, "sphere_centered_0.5"),
    ),
    FixtureGeneratorEntry(
        "turbo_enn_thompson_single_seed",
        "turbo_enn_thompson_single",
        UNIT_BOX_BOUNDS,
        FixtureRunSpec(4, 3, "sphere_centered_0.5"),
    ),
    FixtureGeneratorEntry(
        "turbo_enn_pareto_multi_seed",
        "turbo_enn_pareto_multi",
        UNIT_BOX_BOUNDS,
        FixtureRunSpec(4, 2, "sphere_centered_0.5"),
    ),
    FixtureGeneratorEntry(
        "turbo_zero_seed",
        "turbo_zero",
        UNIT_BOX_BOUNDS,
        FixtureRunSpec(4, 3, "sphere_centered_0.5"),
    ),
    FixtureGeneratorEntry(
        "turbo_enn_noise_aware_seed",
        "turbo_enn_noise_aware",
        UNIT_BOX_BOUNDS,
        FixtureRunSpec(4, 2, "sphere_centered_0.5"),
    ),
    FixtureGeneratorEntry(
        "lhd_only_seed",
        "lhd_only",
        UNIT_BOX_BOUNDS,
        FixtureRunSpec(4, 3, "sphere_centered_0.5"),
    ),
    FixtureGeneratorEntry(
        "morbo_enn_separable_unimodal_seed",
        "morbo_enn_separable_unimodal",
        MORBO_SEPARABLE_BOUNDS,
        FixtureRunSpec(8, 4, "separable_unimodal_two_metric"),
        morbo=True,
    ),
)


def fixture_name_prefix(name: str) -> str:
    m = re.match(r"^(.*)_seed\d+$", name)
    return m.group(1) if m else name


def entry_for_fixture_name(name: str) -> FixtureGeneratorEntry:
    key = fixture_name_prefix(name)
    for entry in FIXTURE_GENERATOR_ENTRIES:
        if entry.config_key == key:
            return entry
    raise ValueError(f"unknown fixture name {name!r}")


def fixture_subdir_for_entry(entry: FixtureGeneratorEntry) -> str:
    return "morbo" if entry.morbo else "python_optimizer"


def catalog_fixture_names() -> tuple[str, ...]:
    return tuple(
        f"{entry.prefix}{seed}"
        for entry in FIXTURE_GENERATOR_ENTRIES
        for seed in (0, 1, 2)
    )


EXPECTED_OPTIMIZER_FIXTURE_NAMES = catalog_fixture_names()
