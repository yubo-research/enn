from __future__ import annotations

from .parity_types import ParityCase, ParityReport


def run_posterior_simple(report: ParityReport) -> None:
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


def run_posterior_observation_noise(report: ParityReport) -> None:
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
