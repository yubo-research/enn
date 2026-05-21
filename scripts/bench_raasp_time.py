from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BenchResult:
    num_candidates: int
    times_s: list[float]
    error: str | None = None


def _parse_int_list(csv: str) -> list[int]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("no candidates provided")
    if any(v <= 0 for v in values):
        raise ValueError(f"all candidates must be > 0, got {values}")
    return values


def _bench_one(
    *,
    num_dim: int,
    num_candidates: int,
    rep_seed: int,
    box: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    from scipy.stats import qmc

    from enn.turbo.config.candidate_rv import CandidateRV
    from enn.turbo.python_fallback.turbo_utils import generate_raasp_candidates

    center, lb, ub = box
    rng = np.random.default_rng(rep_seed)
    t0 = time.perf_counter()
    sobol = qmc.Sobol(d=num_dim, scramble=True, seed=rep_seed)
    x = generate_raasp_candidates(
        center,
        lb,
        ub,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol,
    )
    _ = float(np.sum(x))
    return time.perf_counter() - t0


def _bench_repeats(
    *,
    num_dim: int,
    num_candidates: int,
    repeats: int,
    seed: int,
    box: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[float], str | None]:
    times_s: list[float] = []
    for rep in range(repeats):
        rep_seed = seed + rep
        try:
            times_s.append(
                _bench_one(
                    num_dim=num_dim,
                    num_candidates=num_candidates,
                    rep_seed=rep_seed,
                    box=box,
                )
            )
        except MemoryError:
            return times_s, "MemoryError"
    return times_s, None


def bench_raasp(
    *,
    num_dim: int,
    num_candidates_list: list[int],
    repeats: int,
    seed: int,
) -> list[BenchResult]:
    if num_dim <= 0:
        raise ValueError(num_dim)
    if repeats <= 0:
        raise ValueError(repeats)

    center = np.full(num_dim, 0.5, dtype=float)
    lb = np.zeros(num_dim, dtype=float)
    ub = np.ones(num_dim, dtype=float)

    results: list[BenchResult] = []
    for num_candidates in num_candidates_list:
        times_s, error = _bench_repeats(
            num_dim=num_dim,
            num_candidates=num_candidates,
            repeats=repeats,
            seed=seed,
            box=(center, lb, ub),
        )
        results.append(
            BenchResult(num_candidates=num_candidates, times_s=times_s, error=error)
        )
    return results


def _fmt_stats(times_s: list[float]) -> str:
    if not times_s:
        return "n/a"
    if len(times_s) == 1:
        return f"{times_s[0]:.4f}s"
    return (
        f"min={min(times_s):.4f}s "
        f"median={statistics.median(times_s):.4f}s "
        f"mean={statistics.mean(times_s):.4f}s "
        f"(n={len(times_s)})"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark RAASP candidate generation time."
    )
    p.add_argument("--num-dim", type=int, default=10_000)
    p.add_argument(
        "--candidates",
        type=str,
        default="50,100,200,500,1000,2000",
        help="Comma-separated list of num_candidates values.",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    num_candidates_list = _parse_int_list(args.candidates)
    results = bench_raasp(
        num_dim=int(args.num_dim),
        num_candidates_list=num_candidates_list,
        repeats=int(args.repeats),
        seed=int(args.seed),
    )

    print(f"num_dim={args.num_dim} repeats={args.repeats} seed={args.seed}")
    print("num_candidates\tseconds")
    for r in results:
        if r.error is not None:
            print(f"{r.num_candidates}\t\tERROR: {r.error}")
        else:
            print(f"{r.num_candidates}\t\t{_fmt_stats(r.times_s)}")


if __name__ == "__main__":
    main()
