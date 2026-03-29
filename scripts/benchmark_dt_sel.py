from __future__ import annotations

import argparse
import statistics
import time
from typing import Sequence

import numpy as np

from enn import AcqType, create_optimizer, turbo_enn_config
from enn.turbo.config import ENNFitConfig, ENNSurrogateConfig


def make_bounds(num_dim: int) -> np.ndarray:
    return np.tile(np.array([[0.0, 1.0]], dtype=float), (num_dim, 1))


def synthetic_objective(x: np.ndarray) -> np.ndarray:
    """Cheap objective; shape (n, 1)."""
    # Smooth non-separable signal to avoid degenerate neighborhoods.
    y = -np.sum((x - 0.35) ** 2, axis=1) + 0.1 * np.sin(8.0 * x).sum(axis=1)
    return y.reshape(-1, 1).astype(float)


def seed_observations(
    opt: object,
    rng: np.random.Generator,
    *,
    num_obs: int,
    num_dim: int,
    batch_size: int,
) -> None:
    bounds = make_bounds(num_dim)
    low, high = bounds[:, 0], bounds[:, 1]
    remaining = int(num_obs)
    while remaining > 0:
        n_batch = min(batch_size, remaining)
        x = rng.uniform(low, high, size=(n_batch, num_dim))
        y = synthetic_objective(x)
        opt.tell(x, y)
        remaining -= n_batch


def summarize(values: list[float], name: str) -> None:
    vals_ms = [1000.0 * v for v in values]
    print(
        f"{name}_ms",
        f"mean={statistics.fmean(vals_ms):.3f}",
        f"median={statistics.median(vals_ms):.3f}",
        f"p95={np.percentile(vals_ms, 95):.3f}",
        f"min={min(vals_ms):.3f}",
        f"max={max(vals_ms):.3f}",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stress-test ENN selection timing (dt_sel) at large N, D."
    )
    parser.add_argument(
        "--num-obs", type=int, default=50_000, help="Total observations N"
    )
    parser.add_argument("--num-dim", type=int, default=64, help="Dimension D")
    parser.add_argument("--k", type=int, default=64, help="ENN neighbor count")
    parser.add_argument(
        "--num-arms", type=int, default=4, help="Arms requested per ask()"
    )
    parser.add_argument("--num-fit-samples", type=int, default=64)
    parser.add_argument("--num-fit-candidates", type=int, default=128)
    parser.add_argument("--seed-batch-size", type=int, default=2_000)
    parser.add_argument("--warmup-asks", type=int, default=3)
    parser.add_argument("--timed-asks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _build_optimizer(
    *,
    num_dim: int,
    k: int,
    num_fit_samples: int,
    num_fit_candidates: int,
    num_arms: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    bounds = make_bounds(num_dim)
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(
            k=k,
            fit=ENNFitConfig(
                num_fit_samples=num_fit_samples,
                num_fit_candidates=num_fit_candidates,
            ),
        ),
        num_init=num_arms,
        acq_type=AcqType.UCB,
    )
    return bounds, rng, create_optimizer(bounds=bounds, config=config, rng=rng)


def _run_ask_timings(
    opt, num_arms: int, warmup_asks: int, timed_asks: int
) -> tuple[list[float], list[float], list[float], list[float]]:
    for _ in range(warmup_asks):
        _ = opt.ask(num_arms)

    dt_sel_values: list[float] = []
    dt_gen_values: list[float] = []
    dt_fit_values: list[float] = []
    ask_wall_values: list[float] = []
    for _ in range(timed_asks):
        t0 = time.perf_counter()
        _ = opt.ask(num_arms)
        ask_wall_values.append(time.perf_counter() - t0)
        t = opt.telemetry()
        dt_sel_values.append(float(t.dt_sel))
        dt_gen_values.append(float(t.dt_gen))
        dt_fit_values.append(float(t.dt_fit))
    return dt_sel_values, dt_gen_values, dt_fit_values, ask_wall_values


def _print_setup(num_obs: int, num_dim: int, k: int, num_arms: int):
    print(
        "setup",
        f"N={num_obs}",
        f"D={num_dim}",
        f"k={k}",
        f"num_candidates={min(5000, 100 * num_dim)}",
        f"num_arms={num_arms}",
    )


def run_benchmark(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.num_obs <= 0 or args.num_dim <= 0 or args.k <= 0:
        raise ValueError("num_obs, num_dim, and k must all be positive")

    _, rng, opt = _build_optimizer(
        num_dim=args.num_dim,
        k=args.k,
        num_fit_samples=args.num_fit_samples,
        num_fit_candidates=args.num_fit_candidates,
        num_arms=args.num_arms,
        seed=args.seed,
    )
    _print_setup(
        num_obs=args.num_obs,
        num_dim=args.num_dim,
        k=args.k,
        num_arms=args.num_arms,
    )
    print("seeding observations...")
    t0_seed = time.perf_counter()
    seed_observations(
        opt,
        rng,
        num_obs=args.num_obs,
        num_dim=args.num_dim,
        batch_size=args.seed_batch_size,
    )
    dt_seed = time.perf_counter() - t0_seed
    print(f"seed_time_sec={dt_seed:.3f}")

    dt_sel_values, dt_gen_values, dt_fit_values, ask_wall_values = _run_ask_timings(
        opt,
        args.num_arms,
        args.warmup_asks,
        args.timed_asks,
    )

    print(f"timed_asks={args.timed_asks}")
    summarize(dt_sel_values, "dt_sel")
    summarize(dt_gen_values, "dt_gen")
    summarize(dt_fit_values, "dt_fit")
    summarize(ask_wall_values, "ask_wall")
    ratio = np.mean(dt_sel_values) / np.mean(ask_wall_values)
    print(f"dt_sel_fraction_of_ask_wall={ratio:.3f}")


if __name__ == "__main__":
    run_benchmark()
