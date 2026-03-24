"""Differential parity harness: runs parity checks and produces machine-readable report.

Usage:
  python -m tests.parity.parity_harness
  PYTHONPATH=src python -m tests.parity.parity_harness

Output: writes parity_report.json with per-endpoint pass/fail for CI gating.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ParityCase:
    """Single parity check result."""

    name: str
    endpoint: str
    passed: bool
    error: str | None = None
    backend: str = "rust_vs_python"
    metrics: dict[str, float] | None = None


@dataclass
class ParityReport:
    """Machine-readable parity report for CI gating."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    rust_available: bool = False
    cases: list[ParityCase] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "rust_available": self.rust_available,
            "pct_parity": (self.passed / self.total * 100) if self.total > 0 else 0.0,
            "cases": [asdict(c) for c in self.cases],
        }


def _run_posterior_simple(report: ParityReport) -> None:
    """Posterior: Python public API vs Rust direct."""
    import numpy as np

    from enn import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams, PosteriorFlags

    try:
        from enn._rust import EpistemicNearestNeighbors as RustENN
    except ImportError:
        report.cases.append(
            ParityCase(
                name="posterior_simple",
                endpoint="EpistemicNearestNeighbors.posterior",
                passed=False,
                error="Rust not available",
                backend="rust_vs_python",
            )
        )
        report.skipped += 1
        report.total += 1
        return

    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    params = ENNParams(
        k_num_neighbors=2,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
    )
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=False)

    try:
        model = EpistemicNearestNeighbors(train_x, train_y, scale_x=False)
        out = model.posterior(query, params=params, flags=flags)
    except Exception as e:
        report.cases.append(
            ParityCase(
                name="posterior_simple",
                endpoint="EpistemicNearestNeighbors.posterior",
                passed=False,
                error=str(e),
                backend="rust_vs_python",
            )
        )
        report.failed += 1
        report.total += 1
        return

    try:
        rs_model = RustENN(train_x, train_y, scale_x=False, index_driver="Exact")
        rs_mu, rs_se, _ = rs_model.posterior(
            query,
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
            exclude_nearest=False,
            observation_noise=False,
        )
    except Exception as e:
        report.cases.append(
            ParityCase(
                name="posterior_simple",
                endpoint="EpistemicNearestNeighbors.posterior",
                passed=False,
                error=str(e),
                backend="rust_vs_python",
            )
        )
        report.failed += 1
        report.total += 1
        return

    mu_ok = np.allclose(out.mu, rs_mu, rtol=1e-12, atol=1e-12)
    se_ok = np.allclose(out.se, rs_se, rtol=1e-12, atol=1e-12)
    passed = mu_ok and se_ok and out.idx is not None

    report.cases.append(
        ParityCase(
            name="posterior_simple",
            endpoint="EpistemicNearestNeighbors.posterior",
            passed=bool(passed),
            error=None
            if passed
            else f"mu_ok={mu_ok} se_ok={se_ok} idx={out.idx is not None}",
            backend="rust_vs_python",
        )
    )
    if passed:
        report.passed += 1
    else:
        report.failed += 1
    report.total += 1


def _run_posterior_observation_noise(report: ParityReport) -> None:
    """Posterior with train_yvar and observation_noise."""
    import numpy as np

    from enn import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams, PosteriorFlags

    try:
        from enn._rust import EpistemicNearestNeighbors as RustENN
    except ImportError:
        report.cases.append(
            ParityCase(
                name="posterior_observation_noise",
                endpoint="EpistemicNearestNeighbors.posterior",
                passed=False,
                error="Rust not available",
                backend="rust_vs_python",
            )
        )
        report.skipped += 1
        report.total += 1
        return

    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    train_yvar = np.array([[0.01], [0.01], [0.01], [0.01]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    params = ENNParams(
        k_num_neighbors=2,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
    )
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=True)

    try:
        model = EpistemicNearestNeighbors(
            train_x, train_y, train_yvar=train_yvar, scale_x=False
        )
        out = model.posterior(query, params=params, flags=flags)
    except Exception as e:
        report.cases.append(
            ParityCase(
                name="posterior_observation_noise",
                endpoint="EpistemicNearestNeighbors.posterior",
                passed=False,
                error=str(e),
                backend="rust_vs_python",
            )
        )
        report.failed += 1
        report.total += 1
        return

    try:
        rs_model = RustENN(
            train_x, train_y, train_yvar=train_yvar, scale_x=False, index_driver="Exact"
        )
        rs_mu, rs_se, _ = rs_model.posterior(
            query,
            k_num_neighbors=2,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
            exclude_nearest=False,
            observation_noise=True,
        )
    except Exception as e:
        report.cases.append(
            ParityCase(
                name="posterior_observation_noise",
                endpoint="EpistemicNearestNeighbors.posterior",
                passed=False,
                error=str(e),
                backend="rust_vs_python",
            )
        )
        report.failed += 1
        report.total += 1
        return

    mu_ok = np.allclose(out.mu, rs_mu, rtol=1e-12, atol=1e-12)
    se_ok = np.allclose(out.se, rs_se, rtol=1e-10, atol=1e-10)
    passed = mu_ok and se_ok

    report.cases.append(
        ParityCase(
            name="posterior_observation_noise",
            endpoint="EpistemicNearestNeighbors.posterior",
            passed=bool(passed),
            error=None if passed else f"mu_ok={mu_ok} se_ok={se_ok}",
            backend="rust_vs_python",
        )
    )
    if passed:
        report.passed += 1
    else:
        report.failed += 1
    report.total += 1


def _append_skipped_case(
    report: ParityReport,
    *,
    name: str,
    endpoint: str,
    backend: str,
    reason: str,
) -> None:
    report.cases.append(
        ParityCase(
            name=name,
            endpoint=endpoint,
            passed=False,
            error=reason,
            backend=backend,
        )
    )
    report.skipped += 1
    report.total += 1


def _sobol_abs_error_metrics(
    sobol_sequence_fn,
    scipy_qmc,
    *,
    dims: tuple[int, ...],
    n_points: int,
) -> tuple[float, float]:
    import numpy as np

    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    for d in dims:
        rust_seq = np.asarray(sobol_sequence_fn(d, n_points), dtype=float)
        scipy_seq = scipy_qmc.Sobol(d=d, scramble=False).random_base2(m=6)
        delta = np.abs(rust_seq - scipy_seq)
        max_abs = max(max_abs, float(delta.max()))
        sum_abs += float(delta.sum())
        count += int(delta.size)
    return max_abs, sum_abs / float(count)


def _run_sobol_sequence_parity(report: ParityReport) -> None:
    """Sobol sequence parity: Rust helper vs SciPy qmc.Sobol."""
    try:
        from scipy.stats import qmc
    except ImportError:
        _append_skipped_case(
            report,
            name="sobol_sequence_parity",
            endpoint="util.sobol_sequence",
            backend="rust_vs_scipy",
            reason="SciPy not available",
        )
        return

    try:
        from enn._rust import sobol_sequence
    except ImportError:
        _append_skipped_case(
            report,
            name="sobol_sequence_parity",
            endpoint="util.sobol_sequence",
            backend="rust_vs_scipy",
            reason="Rust not available",
        )
        return

    dims = (2, 3, 5)
    n_points = 64

    try:
        max_abs, mean_abs = _sobol_abs_error_metrics(
            sobol_sequence, qmc, dims=dims, n_points=n_points
        )
    except Exception as e:
        report.cases.append(
            ParityCase(
                name="sobol_sequence_parity",
                endpoint="util.sobol_sequence",
                passed=False,
                error=str(e),
                backend="rust_vs_scipy",
            )
        )
        report.failed += 1
        report.total += 1
        return

    passed = bool(max_abs <= 1e-12)
    report.cases.append(
        ParityCase(
            name="sobol_sequence_parity",
            endpoint="util.sobol_sequence",
            passed=passed,
            error=None if passed else f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}",
            backend="rust_vs_scipy",
            metrics={"max_abs": max_abs, "mean_abs": mean_abs},
        )
    )
    if passed:
        report.passed += 1
    else:
        report.failed += 1
    report.total += 1


def _run_optimizer_parity(
    report: ParityReport,
    name: str,
    config,
) -> None:
    """Run optimizer shape/bounds/state parity for a given config."""
    import numpy as np

    try:
        from enn._rust import Optimizer  # noqa: F401
    except ImportError:
        _append_skipped_case(
            report,
            name=name,
            endpoint="create_optimizer",
            backend="rust",
            reason="Rust not available",
        )
        return

    def _run() -> None:
        from .optimizer_parity_helpers import check_opt_contract, get_rust_optimizer

        bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        opt = get_rust_optimizer(bounds, config, seed=47)
        check_opt_contract(opt, bounds)

    try:
        _run()
    except Exception as e:
        report.cases.append(
            ParityCase(
                name=name,
                endpoint="create_optimizer",
                passed=False,
                error=str(e),
                backend="rust",
            )
        )
        report.failed += 1
    else:
        report.cases.append(
            ParityCase(
                name=name, endpoint="create_optimizer", passed=True, backend="rust"
            )
        )
        report.passed += 1
    report.total += 1


def run_harness() -> ParityReport:
    """Run all parity checks and return report."""
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

    _run_posterior_simple(report)
    _run_posterior_observation_noise(report)
    _run_sobol_sequence_parity(report)
    _run_optimizer_parity(
        report,
        "optimizer_enn_parity",
        turbo_enn_config(
            acq_type=AcqType.UCB,
            enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
            num_init=4,
        ),
    )
    _run_optimizer_parity(
        report,
        "optimizer_zero_parity",
        turbo_zero_config(num_init=4),
    )
    _run_optimizer_parity(
        report,
        "optimizer_lhd_parity",
        lhd_only_config(num_init=5),
    )

    return report


def main() -> int:
    """Entry point: run harness, write report, exit with 0 if all passed."""
    report = run_harness()
    out_path = Path(__file__).resolve().parent.parent.parent / "parity_report.json"
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"Parity report written to {out_path}")
    print(f"Passed: {report.passed}/{report.total}, Failed: {report.failed}")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
