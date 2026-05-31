#!/usr/bin/env python3
"""Compare current ennbo source to ennbo==0.3.6 on TuRBO-ENN, TuRBO-ONE, and MORBO.

Runs three difficult benchmarks (Ackley 30D, DoubleAckley 30D, separable unimodal)
for each optimizer mode and reports answer quality plus wall-clock / ask() timing.

Example:
  python scripts/compare_ennbo_versions.py
  python scripts/compare_ennbo_versions.py --quick
  python scripts/compare_ennbo_versions.py worker --optimizer turbo_enn --problem ackley_30d --version current
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_ennbo_versions_common import (  # noqa: E402
    OPTIMIZER_NAMES,
    PROBLEMS,
    BenchmarkResult,
    apply_quick_overrides,
    experiment_combos,
    run_benchmark,
)

BASELINE_VERSION = "0.3.6"
CURRENT_LABEL = "current"
BASELINE_LABEL = BASELINE_VERSION


def _pythonpath_with_src(env: dict[str, str], src: str) -> None:
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if src not in parts:
        parts.insert(0, src)
    env["PYTHONPATH"] = os.pathsep.join(parts)


def _pythonpath_without_src(env: dict[str, str], src: str) -> None:
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != src]
    if parts:
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return
    env.pop("PYTHONPATH", None)


def _env_for_version(version_label: str) -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    if version_label == CURRENT_LABEL:
        _pythonpath_with_src(env, src)
    else:
        _pythonpath_without_src(env, src)
    return env


def _run_worker_subprocess(
    *,
    optimizer: str,
    problem: str,
    version_label: str,
    quick: bool,
) -> BenchmarkResult:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--optimizer",
        optimizer,
        "--problem",
        problem,
        "--version",
        version_label,
    ]
    if quick:
        cmd.append("--quick")
    proc = subprocess.run(
        cmd,
        env=_env_for_version(version_label),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed ({optimizer}, {problem}, {version_label}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    payload = json.loads(proc.stdout)
    return BenchmarkResult(**payload)


def _worker_main(args: argparse.Namespace) -> int:
    problems = apply_quick_overrides(PROBLEMS) if args.quick else PROBLEMS
    problem = problems[args.problem]
    result = run_benchmark(
        optimizer=args.optimizer,
        problem=problem,
        version_label=args.version,
    )
    print(json.dumps(result.to_dict()))
    return 0


def _format_row(result: BenchmarkResult) -> str:
    return (
        f"{result.version_label:8s}  {result.optimizer:10s}  {result.problem:22s}  "
        f"{result.quality_metric:12s}  {result.quality:12.4f}  "
        f"wall={result.wall_seconds:8.2f}s  ask={result.ask_seconds:8.2f}s"
    )


def _compare_summary(
    current: BenchmarkResult, baseline: BenchmarkResult
) -> dict[str, float | str]:
    quality_delta = current.quality - baseline.quality
    return {
        "optimizer": current.optimizer,
        "problem": current.problem,
        "quality_metric": current.quality_metric,
        "current_quality": current.quality,
        "baseline_quality": baseline.quality,
        "quality_delta": quality_delta,
        "current_wall_s": current.wall_seconds,
        "baseline_wall_s": baseline.wall_seconds,
        "wall_ratio": (
            current.wall_seconds / baseline.wall_seconds
            if baseline.wall_seconds > 0
            else float("inf")
        ),
        "current_ask_s": current.ask_seconds,
        "baseline_ask_s": baseline.ask_seconds,
    }


def _main(args: argparse.Namespace) -> int:
    problems = apply_quick_overrides(PROBLEMS) if args.quick else PROBLEMS
    combos = experiment_combos(problems)

    print("Predicted running time: up to 15 minutes (use --quick for a smoke run)")
    print(
        f"Running {len(combos)} optimizer/problem pairs x 2 versions = {2 * len(combos)} benchmarks\n"
    )

    results: dict[tuple[str, str, str], BenchmarkResult] = {}
    t0 = time.perf_counter()
    for version_label in (CURRENT_LABEL, BASELINE_LABEL):
        print(f"=== ennbo {version_label} ===")
        for optimizer, problem_name in combos:
            label = f"{version_label}:{optimizer}:{problem_name}"
            print(f"  running {label}...", flush=True)
            result = _run_worker_subprocess(
                optimizer=optimizer,
                problem=problem_name,
                version_label=version_label,
                quick=args.quick,
            )
            results[(version_label, optimizer, problem_name)] = result
            print(f"    {_format_row(result)}")
        print()

    comparisons: list[dict[str, float | str]] = []
    print("=== comparison (current minus baseline) ===")
    for optimizer, problem_name in combos:
        current = results[(CURRENT_LABEL, optimizer, problem_name)]
        baseline = results[(BASELINE_LABEL, optimizer, problem_name)]
        summary = _compare_summary(current, baseline)
        comparisons.append(summary)
        print(
            f"{optimizer:10s}  {problem_name:22s}  "
            f"Δquality={summary['quality_delta']:+.4f}  "
            f"wall_ratio={summary['wall_ratio']:.3f}x"
        )

    out_path = ROOT / "compare_ennbo_versions_report.json"
    report = {
        "baseline_version": BASELINE_VERSION,
        "quick": args.quick,
        "elapsed_seconds": time.perf_counter() - t0,
        "results": [r.to_dict() for r in results.values()],
        "comparisons": comparisons,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total elapsed: {report['elapsed_seconds']:.1f}s")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run full comparison (default)")
    run_parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced iteration counts for smoke testing",
    )

    worker_parser = sub.add_parser(
        "worker", help="Run a single benchmark cell (JSON stdout)"
    )
    worker_parser.add_argument("--optimizer", choices=OPTIMIZER_NAMES, required=True)
    worker_parser.add_argument(
        "--problem", choices=tuple(PROBLEMS.keys()), required=True
    )
    worker_parser.add_argument(
        "--version",
        choices=(CURRENT_LABEL, BASELINE_LABEL),
        required=True,
    )
    worker_parser.add_argument("--quick", action="store_true")

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced iteration counts for smoke testing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "worker":
        return _worker_main(args)
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
