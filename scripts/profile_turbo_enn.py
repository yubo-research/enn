from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from dataclasses import dataclass

import numpy as np

from enn import create_optimizer
from enn.turbo.config.candidate_gen_config import (
    CandidateGenConfig,
    const_num_candidates,
)
from enn.turbo.config.enums import AcqType
from enn.turbo.config.enn_surrogate_config import ENNSurrogateConfig, ENNFitConfig
from enn.turbo.config.trust_region import TurboTRConfig
from enn.turbo.config.factory import turbo_enn_config


@dataclass(frozen=True)
class ProfileConfig:
    num_dim: int
    num_obs: int
    num_arms: int
    num_candidates: int | None
    num_fit_samples: int
    num_fit_candidates: int
    seed: int


def _make_bounds(num_dim: int) -> np.ndarray:
    return np.array([[0.0, 1.0]] * num_dim, dtype=float)


def _make_optimizer(cfg: ProfileConfig) -> object:
    rng = np.random.default_rng(cfg.seed)
    bounds = _make_bounds(cfg.num_dim)
    enn_cfg = ENNSurrogateConfig(
        k=10,
        fit=ENNFitConfig(
            num_fit_samples=cfg.num_fit_samples,
            num_fit_candidates=cfg.num_fit_candidates,
        ),
    )
    candidates = (
        CandidateGenConfig(num_candidates=const_num_candidates(cfg.num_candidates))
        if cfg.num_candidates is not None
        else None
    )
    opt_config = turbo_enn_config(
        enn=enn_cfg,
        acq_type=AcqType.UCB,
        trust_region=TurboTRConfig(),
        num_init=1,
        candidates=candidates,
    )
    return create_optimizer(bounds=bounds, config=opt_config, rng=rng)


def _seed_observations(
    opt: object, rng: np.random.Generator, cfg: ProfileConfig
) -> None:
    num_obs = int(cfg.num_obs)
    num_dim = int(cfg.num_dim)
    bounds = _make_bounds(num_dim)
    x = rng.uniform(bounds[:, 0], bounds[:, 1], size=(num_obs, num_dim))
    y = -np.sum(x**2, axis=1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    opt.tell(x, y)
    # Skip init LHD since we've already seeded observations.
    if hasattr(opt, "_strategy") and hasattr(opt._strategy, "_num_init"):
        opt._strategy._init_idx = opt._strategy._num_init


def _time_ask(opt: object, num_arms: int, repeats: int = 3) -> float:
    # Warmup to ensure model is fitted and FAISS index is built
    _ = opt.ask(num_arms=num_arms)
    t_min = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = opt.ask(num_arms=num_arms)
        t_min = min(t_min, time.perf_counter() - t0)
    return t_min


def _time_find_center(opt: object, rng: np.random.Generator) -> float:
    x_obs = opt._x_obs.view()
    y_obs = opt._y_obs.view()
    _ = opt._surrogate.fit(x_obs, y_obs, None, num_steps=0, rng=rng)
    t0 = time.perf_counter()
    _ = opt._find_x_center(x_obs, y_obs)
    return time.perf_counter() - t0


def _time_fit(opt: object, rng: np.random.Generator) -> float:
    x_obs = opt._x_obs.view()
    y_obs = opt._y_obs.view()
    t0 = time.perf_counter()
    _ = opt._surrogate.fit(x_obs, y_obs, None, num_steps=0, rng=rng)
    return time.perf_counter() - t0


def _time_tell(opt: object, rng: np.random.Generator, cfg: ProfileConfig) -> float:
    num_dim = int(cfg.num_dim)
    bounds = _make_bounds(num_dim)
    x = rng.uniform(bounds[:, 0], bounds[:, 1], size=(cfg.num_arms, num_dim))
    y = -np.sum(x**2, axis=1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Profile tell() specifically
    prof = cProfile.Profile()
    prof.enable()
    opt.tell(x, y)
    prof.disable()
    if cfg.num_obs >= 8000:
        print(f"\nProfile for tell() at num_obs={cfg.num_obs}:")
        stats = pstats.Stats(prof).sort_stats("tottime")
        stats.print_stats(20)

    t0 = time.perf_counter()
    opt.tell(x, y)
    return time.perf_counter() - t0


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError((x.shape, y.shape))
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    y_hat = X @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return intercept, slope, r2


def _estimate_scaling(ns: list[int], times: list[float], label: str) -> None:
    n = np.asarray(ns, dtype=float)
    t = np.asarray(times, dtype=float)
    intercept_n, slope_n, r2_n = _ols_fit(n, t)
    intercept_n2, slope_n2, r2_n2 = _ols_fit(n**2, t)
    if r2_n2 > r2_n:
        model = "N^2"
        intercept, slope, r2 = intercept_n2, slope_n2, r2_n2
    else:
        model = "N"
        intercept, slope, r2 = intercept_n, slope_n, r2_n
    print(
        f"{label}_scaling",
        f"model={model}",
        f"intercept={intercept:.6e}",
        f"slope={slope:.6e}",
        f"r2={r2:.6f}",
    )


def run_sweep(cfg: ProfileConfig, *, num_obs_values: list[int]) -> None:
    center_times: list[float] = []
    ask_times: list[float] = []
    fit_times: list[float] = []
    tell_times: list[float] = []
    for n in num_obs_values:
        cfg_n = ProfileConfig(
            num_dim=cfg.num_dim,
            num_obs=int(n),
            num_arms=cfg.num_arms,
            num_candidates=cfg.num_candidates,
            num_fit_samples=cfg.num_fit_samples,
            num_fit_candidates=cfg.num_fit_candidates,
            seed=cfg.seed,
        )
        rng = np.random.default_rng(cfg_n.seed)
        opt = _make_optimizer(cfg_n)
        _seed_observations(opt, rng, cfg_n)
        center_times.append(_time_find_center(opt, rng))
        fit_times.append(_time_fit(opt, rng))
        ask_times.append(_time_ask(opt, cfg_n.num_arms))
        tell_times.append(_time_tell(opt, rng, cfg_n))
        print(
            "sweep",
            "num_obs",
            cfg_n.num_obs,
            "center_time_sec",
            f"{center_times[-1]:.6f}",
            "fit_time_sec",
            f"{fit_times[-1]:.6f}",
            "ask_time_sec",
            f"{ask_times[-1]:.6f}",
            "tell_time_sec",
            f"{tell_times[-1]:.6f}",
        )
    _estimate_scaling(num_obs_values, center_times, "center_time_sec")
    _estimate_scaling(num_obs_values, fit_times, "fit_time_sec")
    _estimate_scaling(num_obs_values, ask_times, "ask_time_sec")
    _estimate_scaling(num_obs_values, tell_times, "tell_time_sec")


def run_profile(cfg: ProfileConfig, *, profile: bool, profile_center: bool) -> None:
    rng = np.random.default_rng(cfg.seed)
    opt = _make_optimizer(cfg)
    _seed_observations(opt, rng, cfg)

    # Warmup to avoid import overhead in profile
    _ = opt.ask(num_arms=cfg.num_arms)

    if profile_center:
        dt_center = _time_find_center(opt, rng)
        print(
            "center_time_sec",
            f"{dt_center:.6f}",
            "num_obs",
            cfg.num_obs,
        )

    if profile:
        prof = cProfile.Profile()
        prof.enable()
        _ = opt.ask(num_arms=cfg.num_arms)
        prof.disable()
        stats = pstats.Stats(prof).sort_stats("tottime")
        stats.print_stats(30)
    else:
        dt = _time_ask(opt, cfg.num_arms)
        print(
            "ask_time_sec",
            f"{dt:.6f}",
            "num_obs",
            cfg.num_obs,
            "num_candidates",
            cfg.num_candidates,
            "num_fit_samples",
            cfg.num_fit_samples,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile TuRBO-ENN ask() with ENN fitting enabled."
    )
    parser.add_argument("--num-dim", type=int, default=8)
    parser.add_argument("--num-obs", type=int, default=2000)
    parser.add_argument("--num-arms", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--num-fit-samples", type=int, default=50)
    parser.add_argument("--num-fit_candidates", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-center", action="store_true")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sweep over num_obs values and estimate scaling.",
    )
    parser.add_argument(
        "--sweep-values",
        type=str,
        default="100,300,1000,3000,10000",
        help="Comma-separated num_obs values for sweep.",
    )
    args = parser.parse_args()

    cfg = ProfileConfig(
        num_dim=args.num_dim,
        num_obs=args.num_obs,
        num_arms=args.num_arms,
        num_candidates=args.num_candidates,
        num_fit_samples=args.num_fit_samples,
        num_fit_candidates=args.num_fit_candidates,
        seed=args.seed,
    )
    if args.sweep:
        sweep_vals = [int(v) for v in args.sweep_values.split(",") if v.strip()]
        if not sweep_vals:
            raise ValueError("--sweep-values must be non-empty")
        run_sweep(cfg, num_obs_values=sweep_vals)
    else:
        run_profile(cfg, profile=args.profile, profile_center=args.profile_center)


if __name__ == "__main__":
    main()
