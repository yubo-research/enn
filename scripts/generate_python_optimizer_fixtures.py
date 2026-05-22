from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from enn import create_optimizer  # noqa: E402
from tests.morbo_objectives import separable_unimodal_objective  # noqa: E402
from enn.turbo.config import (  # noqa: E402
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


def _sphere(x: np.ndarray) -> np.ndarray:
    return (-np.sum((x - 0.5) ** 2, axis=1)).reshape(-1, 1)


def _capture(bounds, config, seed: int, num_cycles: int, num_arms: int) -> dict:
    rng = np.random.default_rng(seed)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    steps = []
    for _ in range(num_cycles):
        x = opt.ask(num_arms=num_arms)
        y = _sphere(x)
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
        "num_cycles": num_cycles,
        "num_arms": num_arms,
        "objective": "sphere_centered_0.5",
        "bounds": bounds.tolist(),
        "steps": steps,
    }


def _enn_ucb(seed: int) -> dict:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 3
    config = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=num_arms,
    )
    return _capture(bounds, config, seed, num_cycles=4, num_arms=num_arms)


def _enn_thompson(seed: int) -> dict:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 3
    config = turbo_enn_config(
        acq_type=AcqType.THOMPSON,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=num_arms,
    )
    return _capture(bounds, config, seed, num_cycles=4, num_arms=num_arms)


def _enn_pareto(seed: int) -> dict:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 2
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=num_arms,
    )
    return _capture(bounds, config, seed, num_cycles=4, num_arms=num_arms)


def _zero(seed: int) -> dict:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 3
    config = turbo_zero_config(num_init=num_arms)
    return _capture(bounds, config, seed, num_cycles=4, num_arms=num_arms)


def _trailing_obs(seed: int) -> dict:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 2
    config = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=num_arms,
        trailing_obs=12,
    )
    return _capture(bounds, config, seed, num_cycles=5, num_arms=num_arms)


def _noise_aware(seed: int) -> dict:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 2
    config = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=8)),
        trust_region=TurboTRConfig(noise_aware=True),
        num_init=num_arms,
    )
    return _capture(bounds, config, seed, num_cycles=4, num_arms=num_arms)


def _lhd(seed: int) -> dict:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 3
    config = lhd_only_config(num_init=num_arms)
    return _capture(bounds, config, seed, num_cycles=4, num_arms=num_arms)


def _morbo_separable_unimodal(seed: int) -> dict:
    bounds = np.array([[-300.0, 600.0], [0.2, 1.0]], dtype=float)
    num_arms = 4
    rng = np.random.default_rng(seed)
    config = turbo_enn_config(
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
    )
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    steps = []
    for _ in range(8):
        x = opt.ask(num_arms=num_arms)
        y = separable_unimodal_objective(x)
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
        "num_cycles": 8,
        "num_arms": num_arms,
        "objective": "separable_unimodal_two_metric",
        "bounds": bounds.tolist(),
        "steps": steps,
    }


def main() -> None:
    out_dir = ROOT / "tests" / "fixtures" / "python_optimizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    morbo_dir = ROOT / "tests" / "fixtures" / "morbo"
    morbo_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("turbo_enn_ucb_single_seed", _enn_ucb),
        ("turbo_enn_thompson_single_seed", _enn_thompson),
        ("turbo_enn_pareto_multi_seed", _enn_pareto),
        ("turbo_zero_seed", _zero),
        ("turbo_enn_trailing_obs_seed", _trailing_obs),
        ("turbo_enn_noise_aware_seed", _noise_aware),
        ("lhd_only_seed", _lhd),
    ]
    for prefix, fn in specs:
        for seed in (0, 1, 2):
            payload = fn(seed)
            path = out_dir / f"{prefix}{seed}.json"
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            print("wrote", path)
    for seed in (0, 1, 2):
        payload = _morbo_separable_unimodal(seed)
        path = morbo_dir / f"morbo_enn_separable_unimodal_seed{seed}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print("wrote", path)


if __name__ == "__main__":
    main()
