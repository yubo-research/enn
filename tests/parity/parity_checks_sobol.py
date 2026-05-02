from __future__ import annotations

from .parity_types import ParityCase, ParityReport


def append_skipped_case(
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


def sobol_abs_error_metrics(
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


def run_sobol_sequence_parity(report: ParityReport) -> None:
    try:
        from scipy.stats import qmc
    except ImportError:
        append_skipped_case(
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
        append_skipped_case(
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
        max_abs, mean_abs = sobol_abs_error_metrics(
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
