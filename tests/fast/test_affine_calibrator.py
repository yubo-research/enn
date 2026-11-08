from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from enn import AffineCalibrator, EpistemicNearestNeighbors, fit_affine_calibrator
from enn.enn.enn_fitter import ENNStatefulFitter
from enn.enn.enn_normal import ENNNormal
from enn.enn.enn_params import ENNParams, PosteriorFlags
from evals.harness.work_dirs import fresh_work_dir


def test_affine_calibrator_ols_recovers_known_map():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=(200, 1))
    a_true, b_true = 0.5, 2.0
    y = a_true + b_true * mu
    se = np.ones_like(mu)
    cal = AffineCalibrator.fit(mu, y, se)
    assert abs(float(cal.a[0]) - a_true) < 1e-9
    assert abs(float(cal.b[0]) - b_true) < 1e-9
    assert abs(float(cal.c[0]) - 0.0) < 1e-9


def test_affine_calibrator_apply_preserves_hypot_identity():
    mu = np.array([[1.0], [2.0], [3.0]], dtype=float)
    se_epi = np.array([[0.4], [0.5], [0.6]], dtype=float)
    se_ale = np.array([[0.3], [0.2], [0.1]], dtype=float)
    se = np.sqrt(se_epi**2 + se_ale**2)
    raw = ENNNormal(mu=mu, se=se, se_epi=se_epi, se_ale=se_ale)
    cal = AffineCalibrator(
        a=np.array([1.0]),
        b=np.array([2.0]),
        c=np.array([0.5]),
    )
    out = cal.apply(raw)
    assert np.allclose(out.mu, 1.0 + 2.0 * mu)
    assert np.allclose(out.se_epi, 0.5 * se_epi)
    assert np.allclose(out.se_ale, 0.5 * se_ale)
    assert np.allclose(out.se, np.sqrt(out.se_epi**2 + out.se_ale**2))


def test_affine_calibrator_degenerate_mu_is_identity():
    mu = np.ones((10, 1), dtype=float)
    y = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
    se = np.ones_like(mu)
    cal = AffineCalibrator.fit(mu, y, se)
    assert float(cal.a[0]) == 0.0
    assert float(cal.b[0]) == 1.0


def test_fit_affine_calibrator_loo_opt_in_on_model():
    rng = np.random.default_rng(1)
    n, d = 40, 3
    x = rng.standard_normal((n, d))
    y = 3.0 + 0.25 * x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    params = ENNParams(
        k_num_neighbors=5,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
    )
    cal = fit_affine_calibrator(
        model, params=params, num_samples=20, seed=int(rng.integers(2**63 - 1))
    )
    assert cal.a.shape == (1,)
    assert cal.b.shape == (1,)
    assert cal.c.shape == (1,)
    assert np.all(np.isfinite(cal.a))
    assert np.all(np.isfinite(cal.b))
    assert np.all(np.isfinite(cal.c))
    assert cal.c[0] > 0.0

    x_q = rng.standard_normal((5, d))
    raw = model.posterior(x_q, params=params)
    adj = cal.apply(raw)
    assert adj.mu.shape == raw.mu.shape
    assert not np.allclose(adj.mu, raw.mu) or abs(float(cal.a[0])) < 1e-6


def test_enn_stateful_fitter_affine_calibrate_flag():
    rng = np.random.default_rng(2)
    n, d = 30, 2
    x = rng.standard_normal((n, d))
    y = 1.0 + 2.0 * x[:, :1]
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    fitter = ENNStatefulFitter(k=4, seed=int(rng.integers(2**63 - 1)))
    params = fitter.ask(
        model,
        num_fit_candidates=8,
        num_fit_samples=12,
        affine_calibrate=False,
    )
    assert params.k_num_neighbors == 4
    assert params.calibrator is None

    params2 = fitter.ask(
        model,
        num_fit_candidates=8,
        num_fit_samples=12,
        affine_calibrate=True,
    )
    assert params2.k_num_neighbors == 4
    assert isinstance(params2.calibrator, AffineCalibrator)


def test_affine_calibrator_exported_from_enn_package():
    import enn

    assert enn.AffineCalibrator is AffineCalibrator
    assert callable(enn.fit_affine_calibrator)


def test_default_posterior_unchanged_without_apply():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((20, 2))
    y = x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    params = ENNParams(
        k_num_neighbors=3,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )
    q = rng.standard_normal((4, 2))
    a = model.posterior(q, params=params)
    b = model.posterior(q, params=params, flags=PosteriorFlags())
    assert np.allclose(a.mu, b.mu)
    assert np.allclose(a.se, b.se)


def test_affine_apply_projects_mu_into_open_y_bounds():
    mu = np.array([[0.2], [0.5], [0.8]], dtype=float)
    se = np.ones_like(mu) * 0.1
    raw = ENNNormal(
        mu=mu,
        se=se,
        se_epi=se.copy(),
        se_ale=np.zeros_like(se),
        y_bounds=np.array([[0.0, 1.0]]),
    )
    cal = AffineCalibrator(
        a=np.array([-0.5]),
        b=np.array([2.0]),
        c=np.array([1.0]),
    )
    out = cal.apply(raw)
    assert np.all(out.mu > 0.0)
    assert np.all(out.mu < 1.0)
    np.random.default_rng(0)
    samples = out.sample(8, seed=0)
    assert np.all(np.isfinite(samples))
    assert np.all(samples > 0.0)
    assert np.all(samples < 1.0)


def test_posterior_applies_calibrator_from_params():
    rng = np.random.default_rng(5)
    n, d = 35, 2
    x = rng.standard_normal((n, d))
    y = 1.0 + 2.0 * x[:, :1]
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    fitter = ENNStatefulFitter(k=4, seed=int(rng.integers(2**63 - 1)))
    params = fitter.ask(
        model,
        num_fit_candidates=8,
        num_fit_samples=12,
        affine_calibrate=True,
    )
    q = x[:4]
    raw = model.posterior(q, params=replace(params, calibrator=None))
    calibrated = model.posterior(q, params=params)
    manual = params.calibrator.apply(raw)
    assert np.allclose(calibrated.mu, manual.mu)
    assert np.allclose(calibrated.se, manual.se)
    assert not np.allclose(calibrated.mu, raw.mu) or (
        abs(float(params.calibrator.a[0])) < 1e-9
        and abs(float(params.calibrator.b[0]) - 1.0) < 1e-9
    )


def test_posterior_passthrough_without_calibrator():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((20, 2))
    y = x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    fitter = ENNStatefulFitter(k=3, seed=int(rng.integers(2**63 - 1)))
    params = fitter.ask(
        model,
        num_fit_candidates=6,
        num_fit_samples=8,
        affine_calibrate=False,
    )
    assert params.calibrator is None
    q = x[:3]
    plain = model.posterior(q, params=ENNParams(3, *_scales(params)))
    via = model.posterior(q, params=params)
    np.testing.assert_array_equal(via.mu, plain.mu)
    np.testing.assert_array_equal(via.se, plain.se)


def _scales(params) -> tuple[float, float]:
    return params.epistemic_variance_scale, params.aleatoric_variance_scale


@pytest.mark.parametrize("n", [0, 1])
def test_fit_affine_calibrator_small_n_returns_identity(n: int):
    rng = np.random.default_rng(4)
    d = 2
    x = rng.standard_normal((n, d)) if n else np.zeros((0, d))
    y = np.zeros((n, 1), dtype=float)
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    params = ENNParams(
        k_num_neighbors=2,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )
    cal = fit_affine_calibrator(
        model, params=params, num_samples=5, seed=int(rng.integers(2**63 - 1))
    )
    assert np.allclose(cal.a, 0.0)
    assert np.allclose(cal.b, 1.0)
    assert np.allclose(cal.c, 1.0)


def test_fit_affine_calibrator_negative_samples_match_zero():
    x = np.random.default_rng(5).standard_normal((12, 2))
    model = EpistemicNearestNeighbors(x, x[:, :1], work_dir=fresh_work_dir())
    params = ENNParams(
        k_num_neighbors=2,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )
    negative, zero = (
        fit_affine_calibrator(
            model,
            params=params,
            num_samples=n,
            seed=int(np.random.default_rng(6).integers(2**63 - 1)),
        )
        for n in (-3, 0)
    )
    assert np.array_equal(negative.a, zero.a)
    assert np.array_equal(negative.b, zero.b)
    assert np.array_equal(negative.c, zero.c)


def test_function_draw_applies_calibrator_from_params():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((40, 2))
    y = 1.0 + 2.0 * x[:, :1]
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    fitter = ENNStatefulFitter(k=4, seed=int(rng.integers(2**63 - 1)))
    params = fitter.ask(
        model, num_fit_candidates=8, num_fit_samples=16, affine_calibrate=True
    )
    q = x[:6]
    bare = replace(params, calibrator=None)
    raw_draws, _ = model.posterior_function_draw(
        q, params=bare, function_seeds=[0, 1, 2]
    )
    cal_draws, _ = model.posterior_function_draw(
        q, params=params, function_seeds=[0, 1, 2]
    )
    raw_mu = model.posterior(q, params=bare).mu
    cal = params.calibrator
    manual = cal.map_draws(np.transpose(raw_draws, (0, 2, 1)), raw_mu)
    manual = np.transpose(manual, (0, 2, 1))
    assert np.allclose(cal_draws, manual)
    assert not np.allclose(cal_draws, raw_draws) or (
        abs(float(cal.a[0])) < 1e-9
        and abs(float(cal.b[0]) - 1.0) < 1e-9
        and abs(float(cal.c[0]) - 1.0) < 1e-9
    )


def test_function_draw_scatter_follows_c_not_b():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((50, 2))
    y = 1.0 + 0.5 * x[:, :1]
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    fitter = ENNStatefulFitter(k=5, seed=int(rng.integers(2**63 - 1)))
    params = fitter.ask(
        model, num_fit_candidates=8, num_fit_samples=16, affine_calibrate=False
    )
    cal = AffineCalibrator(
        a=np.array([0.0]),
        b=np.array([3.0]),
        c=np.array([0.5]),
    )
    q = x[:20]
    seeds = list(range(64))
    raw_d, _ = model.posterior_function_draw(q, params=params, function_seeds=seeds)
    cal_d, _ = model.posterior_function_draw(
        q, params=replace(params, calibrator=cal), function_seeds=seeds
    )
    raw_std = float(np.std(raw_d, axis=-1).mean())
    cal_std = float(np.std(cal_d, axis=-1).mean())
    ratio = cal_std / raw_std
    assert abs(ratio - 0.5) < 0.05
    assert abs(ratio - 3.0) > 1.0
    raw_mu = model.posterior(q, params=params).mu
    mean_cal = cal_d.mean(axis=-1)
    expected_mean = cal.map_mu(raw_mu)
    assert np.allclose(mean_cal, expected_mean, atol=0.05)


def test_apply_coeff_length_is_rust_invalid_shape():
    mu = np.ones((3, 1), dtype=float)
    se = np.ones_like(mu)
    raw = ENNNormal(mu=mu, se=se, se_epi=se.copy(), se_ale=np.zeros_like(se))
    cal = AffineCalibrator(a=np.zeros(2), b=np.ones(2), c=np.ones(2))
    with pytest.raises(ValueError, match="Invalid shape"):
        cal.apply(raw)


def test_affine_unbounded_apply_matches_formula():
    mu = np.array([[0.2], [0.5], [0.8]], dtype=float)
    se = np.ones_like(mu) * 0.1
    raw = ENNNormal(mu=mu, se=se, se_epi=se.copy(), se_ale=np.zeros_like(se))
    cal = AffineCalibrator(
        a=np.array([-0.5]),
        b=np.array([2.0]),
        c=np.array([1.0]),
    )
    out = cal.apply(raw)
    assert np.allclose(out.mu, -0.5 + 2.0 * mu)
