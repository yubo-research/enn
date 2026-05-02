"""Differential parity harness: runs parity checks and produces machine-readable report.

Usage:
  python -m tests.parity.parity_harness
  PYTHONPATH=src python -m tests.parity.parity_harness

Output: writes parity_report.json with per-endpoint pass/fail for CI gating.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .parity_checks_optimizer import run_optimizer_parity
from .parity_checks_posterior import (
    run_posterior_observation_noise,
    run_posterior_simple,
)
from .parity_checks_sobol import run_sobol_sequence_parity
from .parity_types import ParityReport


def run_harness() -> ParityReport:
    from enn.turbo.config import (
        AcqType,
        ENNFitConfig,
        ENNSurrogateConfig,
        lhd_only_config,
        turbo_enn_config,
        turbo_zero_config,
    )

    report = ParityReport()
    try:
        from enn._rust import EpistemicNearestNeighbors as _  # noqa: F401

        report.rust_available = True
    except ImportError:
        report.rust_available = False

    run_posterior_simple(report)
    run_posterior_observation_noise(report)
    run_sobol_sequence_parity(report)
    run_optimizer_parity(
        report,
        "optimizer_enn_parity",
        turbo_enn_config(
            acq_type=AcqType.UCB,
            enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
            num_init=4,
        ),
    )
    run_optimizer_parity(
        report,
        "optimizer_zero_parity",
        turbo_zero_config(num_init=4),
    )
    run_optimizer_parity(
        report,
        "optimizer_lhd_parity",
        lhd_only_config(num_init=5),
    )

    return report


def main() -> int:
    report = run_harness()
    out_path = Path(__file__).resolve().parent.parent.parent / "parity_report.json"
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"Parity report written to {out_path}")
    print(f"Passed: {report.passed}/{report.total}, Failed: {report.failed}")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
