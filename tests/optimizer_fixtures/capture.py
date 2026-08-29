from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from enn.turbo.config.optimizer_config import OptimizerConfig

from .catalog import (
    FIXTURE_OBJECTIVES,
    PREFIX_CONFIG,
    FixtureGeneratorEntry,
    FixtureRunSpec,
    fixture_subdir_for_entry,
)


def capture_optimizer_fixture(
    bounds: np.ndarray,
    config: OptimizerConfig,
    seed: int,
    run: FixtureRunSpec,
) -> dict[str, Any]:
    from enn import create_optimizer

    objective_fn = FIXTURE_OBJECTIVES[run.objective]
    rng = np.random.default_rng(seed)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    steps = []
    for _ in range(run.num_cycles):
        x = opt.ask(num_arms=run.num_arms)
        y = objective_fn(x)
        opt.tell(x, y)
        steps.append(
            {
                "ask": x.tolist(),
                "tell_y": y.tolist(),
                "tr_length": float(opt.tr_length),
                "tr_obs_count": int(opt.tr_obs_count),
            }
        )
    return {
        "seed": seed,
        "num_cycles": run.num_cycles,
        "num_arms": run.num_arms,
        "objective": run.objective,
        "bounds": bounds.tolist(),
        "steps": steps,
    }


def build_fixture(entry: FixtureGeneratorEntry, seed: int) -> dict[str, Any]:
    config = PREFIX_CONFIG[entry.config_key]
    return capture_optimizer_fixture(entry.bounds, config, seed, entry.run)


def fixture_output_path(entry: FixtureGeneratorEntry, seed: int, root: Path) -> Path:
    subdir = fixture_subdir_for_entry(entry)
    return root / "tests" / "fixtures" / subdir / f"{entry.prefix}{seed}.json"
