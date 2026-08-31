"""Tests for AffineCalibrator and optional fitter wiring."""

from __future__ import annotations

import numpy as np

from enn import AffineCalibrator, ENNStatefulFitter, EpistemicNearestNeighbors
from enn.enn.enn_normal import ENNNormal
from enn.enn.enn_params import ENNParams, PosteriorFlags


def test_affine_calibrator_shrinkage_recovers_b():
    """Claim: when mu = y/2 exactly, OLS recovers a≈0, b≈2 and c≈0."""
    y = np.linspace(-1.0, 1.0, 50).reshape(-1, 1)
    mu = 0.5 * y
    se = np.full_like(mu, 0.5 * float(np.std(y)))
    cal = AffineCalibrator.identity(1)
    a, b = cal.fit(mu, y)
    c = cal.fit_residual_scale(mu, se, y)
    assert abs(float(a[0])) < 1e-10
    assert abs(float(b[0]) - 2.0) < 1e-10
    assert float(c[0]) < 1e-10


def test_affine_calibrator_apply_preserves_hypot():
    mu = np.array([[1.0, 2.0]], dtype=float)
    se_epi = np.array([[0.3, 0.4]], dtype=float)
    se_ale = np.array([[0.1, 0.2]], dtype=float)
    se = np.hypot(se_epi, se_ale)
    raw = ENNNormal(mu=mu, se=se, se_epi=se_epi, se_ale=se_ale)
    cal = AffineCalibrator(
        a=np.array([0.5, -0.25]),
        b=np.array([1.5, 0.8]),
        c=np.array([0.7, 1.2]),
    )
    out = cal.apply(raw)
    np.testing.assert_allclose(out.mu, cal.a + cal.b * mu)
    np.testing.assert_allclose(out.se_epi, cal.c * se_epi)
    np.testing.assert_allclose(out.se_ale, cal.c * se_ale)
    np.testing.assert_allclose(out.se, np.hypot(out.se_epi, out.se_ale))


def test_affine_calibrator_rls_matches_batch():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=(40, 2))
    y = 0.3 + 1.7 * mu + rng.normal(scale=0.05, size=mu.shape)
    batch = AffineCalibrator.identity(2)
    batch.fit(mu, y)
    online = AffineCalibrator.identity(2)
    for i in range(mu.shape[0]):
        online.update(mu[i : i + 1], y[i : i + 1])
    np.testing.assert_allclose(online.a, batch.a, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(online.b, batch.b, rtol=1e-10, atol=1e-10)


def test_affine_calibrator_incremental_c_matches_batch():
    """Claim: with se in update(), online c matches batch fit_residual_scale (moments)."""
    rng = np.random.default_rng(2)
    mu = rng.normal(size=(60, 2))
    y = -0.2 + 1.3 * mu + rng.normal(scale=0.08, size=mu.shape)
    se = np.abs(rng.normal(size=mu.shape)) + 0.05
    batch = AffineCalibrator.identity(2)
    batch.fit(mu, y)
    batch.fit_residual_scale(mu, se, y)
    online = AffineCalibrator.identity(2)
    for i in range(mu.shape[0]):
        online.update(mu[i : i + 1], y[i : i + 1], se=se[i : i + 1])
    np.testing.assert_allclose(online.a, batch.a, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(online.b, batch.b, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(online.c, batch.c, rtol=1e-10, atol=1e-10)


def test_fit_residual_scale_without_prior_fit():
    mu = np.array([[0.0], [1.0], [2.0]], dtype=float)
    y = np.array([[0.0], [2.0], [4.0]], dtype=float)
    se = np.ones_like(mu)
    cal = AffineCalibrator.identity(1)
    c = cal.fit_residual_scale(mu, se, y)
    assert cal._n == 3.0
    assert abs(float(cal.b[0]) - 2.0) < 1e-8
    assert float(c[0]) < 1e-8


def test_update_expands_metric_width_and_singular_mu():
    cal = AffineCalibrator.identity(1)
    cal.update(
        np.array([[0.1, -0.2]], dtype=float),
        np.array([[0.3, 0.4]], dtype=float),
        se=np.array([[0.5, 0.6]], dtype=float),
    )
    assert cal.a.shape == (2,)
    cal2 = AffineCalibrator.identity(1)
    mu = np.ones((6, 1), dtype=float)
    y = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    se = np.full_like(mu, 0.2)
    for i in range(mu.shape[0]):
        cal2.update(mu[i : i + 1], y[i : i + 1], se=se[i : i + 1])
    assert float(cal2.b[0]) == 1.0
    assert np.isfinite(float(cal2.c[0]))


def test_fitter_affine_calibrate_optional_default_off():
    train_x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=float,
    )
    train_y = np.array([[0.0], [1.0], [1.0], [2.0], [1.0]], dtype=float)
    model = EpistemicNearestNeighbors(train_x, train_y, scale_x=False)
    rng = np.random.default_rng(7)
    fitter = ENNStatefulFitter(k=2, rng=rng)
    fitter.tell(train_x, train_y)
    params = fitter.ask(model, num_fit_candidates=3, num_fit_samples=4)
    assert isinstance(params, ENNParams)
    assert fitter.affine_calibrator is None


def test_fitter_affine_calibrate_opt_in_metric_gates():
    from evals.flat_sphere import gaussian_loglik
    from ops.qa import nrmse, rcorr

    rng = np.random.default_rng(11)
    n, d = 60, 3
    train_x = rng.uniform(0.0, 1.0, size=(n, d))
    train_y = (train_x.sum(axis=1) + 0.75).reshape(-1, 1)
    model = EpistemicNearestNeighbors(train_x, train_y, scale_x=False)
    fitter = ENNStatefulFitter(k=5, rng=rng, affine_calibrate=True)
    fitter.tell(train_x, train_y)
    params = fitter.ask(model, num_fit_candidates=8, num_fit_samples=20)
    cal = fitter.affine_calibrator
    assert cal is not None

    flags = PosteriorFlags(exclude_nearest=True, observation_noise=True)
    idx = rng.choice(n, size=20, replace=False)
    x_loo, y_loo, _ = model.train_rows_at(idx)
    raw = model.posterior(x_loo, params=params, flags=flags)
    out = cal.apply(raw)
    assert nrmse(y_loo, out.mu) <= nrmse(y_loo, raw.mu) + 1e-12
    assert gaussian_loglik(y_loo, out.mu, out.se) >= gaussian_loglik(
        y_loo, raw.mu, raw.se
    ) - 1e-9
    assert rcorr(y_loo, out.mu) >= rcorr(y_loo, raw.mu) - 1e-9
    np.testing.assert_allclose(out.se, np.hypot(out.se_epi, out.se_ale), rtol=1e-10)
