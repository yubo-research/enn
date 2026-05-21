from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_dt_sel import run_benchmark  # noqa: E402


def assert_ci_speed_gate() -> None:
    baseline = ROOT / "tests" / "fixtures" / "optimizer_speed_baseline.json"
    argv = [
        "--num-obs",
        "100",
        "--num-dim",
        "2",
        "--k",
        "10",
        "--num-arms",
        "1",
        "--timed-asks",
        "5",
        "--warmup-asks",
        "1",
        "--compare-baseline",
        str(baseline),
        "--fail-on-regression",
    ]
    run_benchmark(argv)
