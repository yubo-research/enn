from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

BenchmarkObjective = Callable[[np.ndarray], np.ndarray]


def _const_num_candidates_fn(n: int):
    value = int(n)

    def fn(*, num_dim: int, num_arms: int) -> int:
        return value

    return fn


def _import_optimizer_configs() -> dict[str, Any]:
    try:
        from enn.turbo.config import (
            AcqType,
            CandidateGenConfig,
            ENNFitConfig,
            ENNSurrogateConfig,
            MorboTRConfig,
            MultiObjectiveConfig,
            TurboTRConfig,
            turbo_enn_config,
            turbo_one_config,
        )
    except ImportError:
        from enn.turbo.optimizer_config import (
            AcqType,
            CandidateGenConfig,
            ENNFitConfig,
            ENNSurrogateConfig,
            MorboTRConfig,
            MultiObjectiveConfig,
            TurboTRConfig,
            turbo_enn_config,
            turbo_one_config,
        )
    return {
        "AcqType": AcqType,
        "CandidateGenConfig": CandidateGenConfig,
        "ENNFitConfig": ENNFitConfig,
        "ENNSurrogateConfig": ENNSurrogateConfig,
        "MorboTRConfig": MorboTRConfig,
        "MultiObjectiveConfig": MultiObjectiveConfig,
        "TurboTRConfig": TurboTRConfig,
        "turbo_enn_config": turbo_enn_config,
        "turbo_one_config": turbo_one_config,
    }


def _make_candidate_gen_config(num_candidates: int | None = None) -> Any:
    cfg = _import_optimizer_configs()
    CandidateGenConfig = cfg["CandidateGenConfig"]
    if num_candidates is None:
        return CandidateGenConfig()
    probe = CandidateGenConfig()
    if callable(getattr(probe, "num_candidates", None)):
        try:
            from enn.turbo.config.num_candidates_fn import const_num_candidates
        except ImportError:
            const_num_candidates = _const_num_candidates_fn

        return CandidateGenConfig(num_candidates=const_num_candidates(num_candidates))
    return CandidateGenConfig(num_candidates=num_candidates)


def separable_unimodal_objective(x: np.ndarray) -> np.ndarray:
    try:
        from enn.benchmarks.separable_unimodal import (
            separable_unimodal_objective as objective,
        )

        return objective(x)
    except ImportError:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        c0, c1 = 120.0, 0.91
        y1 = 500_000.0 - 8.0 * (x[:, 0] - c0) ** 2
        y2 = 12.5 - 110.0 * (x[:, 1] - c1) ** 2
        return np.column_stack((y1, y2))


def compute_hypervolume(y: np.ndarray, ref_point: np.ndarray) -> float:
    from enn.enn.enn_util import pareto_front_2d_maximize
    from enn.turbo.hypervolume import hypervolume_2d_max

    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0
    pareto_idx = pareto_front_2d_maximize(y[:, 0], y[:, 1])
    if pareto_idx.size == 0:
        return 0.0
    return float(hypervolume_2d_max(y[pareto_idx], ref_point))


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    num_dim: int
    num_metrics: int
    num_iterations: int
    num_arms: int
    noise: float
    rng_seed: int
    torch_seed: int
    ref_point: tuple[float, float] | None = None

    def bounds(self) -> np.ndarray:
        if self.name == "separable_unimodal":
            return np.array([[-300.0, 600.0], [0.2, 1.0]], dtype=float)
        from enn.benchmarks import Ackley, DoubleAckley

        rng = np.random.default_rng(self.rng_seed)
        if self.name == "ackley_30d":
            objective = Ackley(noise=self.noise, rng=rng)
        elif self.name in {"double_ackley_30d", "ackley_pair_30d"}:
            objective = (
                Ackley(noise=self.noise, rng=rng)
                if self.name == "ackley_pair_30d"
                else DoubleAckley(noise=self.noise, rng=rng)
            )
        else:
            raise ValueError(self.name)
        return np.array([objective.bounds] * self.num_dim, dtype=float)

    def make_objective(self) -> BenchmarkObjective:
        if self.name == "separable_unimodal":
            return separable_unimodal_objective

        from enn.benchmarks import Ackley, DoubleAckley

        rng = np.random.default_rng(self.rng_seed)
        if self.name == "ackley_30d":
            ackley = Ackley(noise=self.noise, rng=rng)

            def objective(x: np.ndarray) -> np.ndarray:
                y = ackley(x)
                return y.reshape(-1, 1) if np.ndim(y) == 1 else y.reshape(-1, 1)

            return objective

        if self.name == "double_ackley_30d":
            double_ackley = DoubleAckley(noise=self.noise, rng=rng)
            return double_ackley

        if self.name == "ackley_pair_30d":
            ackley = Ackley(noise=self.noise, rng=rng)
            mid = self.num_dim // 2

            def objective(x: np.ndarray) -> np.ndarray:
                from enn.benchmarks.ackley_core import ackley_core

                x = np.asarray(x, dtype=float)
                if x.ndim == 1:
                    x = x[None, :]
                n = x.shape[0]
                y1 = -ackley_core(x[:, :mid]) + self.noise * rng.normal(size=n)
                y2 = -ackley_core(x[:, mid:]) + self.noise * rng.normal(size=n)
                return np.stack([y1, y2], axis=1)

            return objective

        raise ValueError(self.name)

    def quality_metric(self) -> str:
        return "hypervolume" if self.num_metrics > 1 else "best_y"


PROBLEMS: dict[str, ProblemSpec] = {
    "ackley_30d": ProblemSpec(
        name="ackley_30d",
        num_dim=30,
        num_metrics=1,
        num_iterations=100,
        num_arms=10,
        noise=0.1,
        rng_seed=18,
        torch_seed=17,
    ),
    "double_ackley_30d": ProblemSpec(
        name="double_ackley_30d",
        num_dim=30,
        num_metrics=2,
        num_iterations=100,
        num_arms=10,
        noise=0.1,
        rng_seed=18,
        torch_seed=17,
        ref_point=(-25.0, -25.0),
    ),
    "separable_unimodal": ProblemSpec(
        name="separable_unimodal",
        num_dim=2,
        num_metrics=2,
        num_iterations=60,
        num_arms=4,
        noise=0.0,
        rng_seed=42,
        torch_seed=17,
        ref_point=(0.0, 0.0),
    ),
    "ackley_pair_30d": ProblemSpec(
        name="ackley_pair_30d",
        num_dim=30,
        num_metrics=2,
        num_iterations=100,
        num_arms=10,
        noise=0.1,
        rng_seed=18,
        torch_seed=17,
        ref_point=(-25.0, -25.0),
    ),
}

OPTIMIZER_NAMES: tuple[str, ...] = ("turbo_enn", "turbo_one", "morbo")

EXPERIMENT_GRID: dict[str, tuple[str, ...]] = {
    "turbo_enn": ("ackley_30d", "double_ackley_30d", "separable_unimodal"),
    "turbo_one": ("ackley_30d", "double_ackley_30d", "separable_unimodal"),
    "morbo": ("double_ackley_30d", "separable_unimodal", "ackley_pair_30d"),
}


def experiment_combos(
    problems: dict[str, ProblemSpec] | None = None,
) -> list[tuple[str, str]]:
    catalog = problems or PROBLEMS
    combos: list[tuple[str, str]] = []
    for optimizer in OPTIMIZER_NAMES:
        for problem_name in EXPERIMENT_GRID[optimizer]:
            combos.append((optimizer, problem_name))
            if problem_name not in catalog:
                raise KeyError(problem_name)
    return combos


def build_optimizer_config(optimizer: str, problem: ProblemSpec) -> Any:
    if optimizer not in OPTIMIZER_NAMES:
        raise ValueError(optimizer)
    cfg = _import_optimizer_configs()
    AcqType = cfg["AcqType"]
    ENNFitConfig = cfg["ENNFitConfig"]
    ENNSurrogateConfig = cfg["ENNSurrogateConfig"]
    MorboTRConfig = cfg["MorboTRConfig"]
    MultiObjectiveConfig = cfg["MultiObjectiveConfig"]
    TurboTRConfig = cfg["TurboTRConfig"]
    turbo_enn_config = cfg["turbo_enn_config"]
    turbo_one_config = cfg["turbo_one_config"]

    turbo_tr = TurboTRConfig(noise_aware=True)
    morbo_tr = None
    if problem.num_metrics > 1:
        multi_objective = MultiObjectiveConfig(num_metrics=problem.num_metrics)
        morbo_tr = MorboTRConfig(multi_objective=multi_objective, noise_aware=True)

    if optimizer == "morbo":
        if morbo_tr is None:
            raise ValueError(f"MORBO requires num_metrics >= 2, got {problem.name}")
        return turbo_enn_config(
            enn=ENNSurrogateConfig(
                k=10,
                fit=ENNFitConfig(num_fit_samples=100),
            ),
            trust_region=morbo_tr,
            acq_type=AcqType.UCB,
            candidates=_make_candidate_gen_config(),
            num_init=min(20, 2 * problem.num_dim),
        )

    if optimizer == "turbo_enn":
        trust_region = morbo_tr if morbo_tr is not None else turbo_tr
        acq_type = AcqType.UCB if problem.num_metrics == 1 else AcqType.THOMPSON
        candidates = (
            _make_candidate_gen_config(64)
            if problem.name == "separable_unimodal"
            else _make_candidate_gen_config()
        )
        return turbo_enn_config(
            enn=ENNSurrogateConfig(
                k=10,
                fit=ENNFitConfig(num_fit_samples=100),
            ),
            trust_region=trust_region,
            acq_type=acq_type,
            candidates=candidates,
            num_init=min(20, 2 * problem.num_dim),
        )

    if optimizer == "turbo_one":
        trust_region = morbo_tr if morbo_tr is not None else turbo_tr
        return turbo_one_config(
            trust_region=trust_region,
            num_init=min(20, 2 * problem.num_dim),
        )

    raise ValueError(optimizer)


@dataclass
class BenchmarkResult:
    optimizer: str
    problem: str
    version_label: str
    quality: float
    quality_metric: str
    wall_seconds: float
    ask_seconds: float
    num_evals: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_benchmark(
    *,
    optimizer: str,
    problem: ProblemSpec,
    version_label: str,
) -> BenchmarkResult:
    import torch

    from enn import create_optimizer

    torch.manual_seed(problem.torch_seed)
    rng = np.random.default_rng(problem.rng_seed)
    bounds = problem.bounds()
    objective = problem.make_objective()
    config = build_optimizer_config(optimizer, problem)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)

    best_y = -np.inf
    all_y: list[np.ndarray] = []
    ask_seconds = 0.0
    num_evals = 0
    t_start = time.perf_counter()

    for _ in range(problem.num_iterations):
        t_ask = time.perf_counter()
        x_arms = opt.ask(num_arms=problem.num_arms)
        ask_seconds += time.perf_counter() - t_ask

        y_obs = objective(x_arms)
        if y_obs.ndim == 1:
            y_obs = y_obs.reshape(-1, 1)
        opt.tell(x_arms, y_obs)
        num_evals += problem.num_arms

        if problem.num_metrics == 1:
            best_y = max(best_y, float(np.max(y_obs)))
        else:
            all_y.append(y_obs)

    wall_seconds = time.perf_counter() - t_start

    if problem.num_metrics == 1:
        quality = float(best_y)
    else:
        ref = np.asarray(problem.ref_point, dtype=float)
        quality = compute_hypervolume(np.vstack(all_y), ref)

    return BenchmarkResult(
        optimizer=optimizer,
        problem=problem.name,
        version_label=version_label,
        quality=quality,
        quality_metric=problem.quality_metric(),
        wall_seconds=wall_seconds,
        ask_seconds=ask_seconds,
        num_evals=num_evals,
        seed=problem.rng_seed,
    )


def apply_quick_overrides(problems: dict[str, ProblemSpec]) -> dict[str, ProblemSpec]:
    return {
        name: ProblemSpec(
            name=spec.name,
            num_dim=spec.num_dim,
            num_metrics=spec.num_metrics,
            num_iterations=max(5, spec.num_iterations // 10),
            num_arms=min(4, spec.num_arms),
            noise=spec.noise,
            rng_seed=spec.rng_seed,
            torch_seed=spec.torch_seed,
            ref_point=spec.ref_point,
        )
        for name, spec in problems.items()
    }
