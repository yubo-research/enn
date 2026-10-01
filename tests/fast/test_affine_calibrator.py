from __future__ import annotations

import numpy as np
import pytest

from enn import AffineCalibrator, EpistemicNearestNeighbors, fit_affine_calibrator
from enn.enn.enn_fitter import ENNStatefulFitter
from enn.enn.enn_normal import ENNNormal
from enn.enn.enn_params import ENNParams, PosteriorFlags


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
    model = EpistemicNearestNeighbors(x, y)
    params = ENNParams(
        k_num_neighbors=5,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
    )
    cal = fit_affine_calibrator(model, params, num_samples=20, rng=rng)
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
    model = EpistemicNearestNeighbors(x, y)
    fitter = ENNStatefulFitter(k=4, rng=rng)
    fitter.tell(x, y)
    assert fitter.affine_calibrator is None
    params = fitter.ask(
        model,
        num_fit_candidates=8,
        num_fit_samples=12,
        affine_calibrate=False,
    )
    assert params.k_num_neighbors == 4
    assert fitter.affine_calibrator is None

    params2 = fitter.ask(
        model,
        num_fit_candidates=8,
        num_fit_samples=12,
        affine_calibrate=True,
    )
    assert params2.k_num_neighbors == 4
    cal = fitter.affine_calibrator
    assert cal is not None
    assert isinstance(cal, AffineCalibrator)


def test_affine_calibrator_exported_from_enn_package():
    import enn

    assert enn.AffineCalibrator is AffineCalibrator
    assert callable(enn.fit_affine_calibrator)


def test_default_posterior_unchanged_without_apply():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((20, 2))
    y = x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(x, y)
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


def test_fitter_posterior_applies_calibrator_when_opted_in():
    rng = np.random.default_rng(5)
    n, d = 35, 2
    x = rng.standard_normal((n, d))
    y = 1.0 + 2.0 * x[:, :1]
    model = EpistemicNearestNeighbors(x, y)
    fitter = ENNStatefulFitter(k=4, rng=rng)
    fitter.tell(x, y)
    params = fitter.ask(
        model,
        num_fit_candidates=8,
        num_fit_samples=12,
        affine_calibrate=True,
    )
    q = x[:4]
    raw = model.posterior(q, params=params)
    via_fitter = fitter.posterior(model, q, params)
    manual = fitter.calibrate(raw)
    assert np.allclose(via_fitter.mu, manual.mu)
    assert np.allclose(via_fitter.se, manual.se)
    assert not np.allclose(via_fitter.mu, raw.mu) or (
        abs(float(fitter.affine_calibrator.a[0])) < 1e-9
        and abs(float(fitter.affine_calibrator.b[0]) - 1.0) < 1e-9
    )


def test_fitter_posterior_passthrough_without_calibrator():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((20, 2))
    y = x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(x, y)
    fitter = ENNStatefulFitter(k=3, rng=rng)
    fitter.tell(x, y)
    params = fitter.ask(
        model,
        num_fit_candidates=6,
        num_fit_samples=8,
        affine_calibrate=False,
    )
    q = x[:3]
    raw = model.posterior(q, params=params)
    via = fitter.posterior(model, q, params)
    assert np.allclose(via.mu, raw.mu)
    assert np.allclose(via.se, raw.se)


@pytest.mark.parametrize("n", [0, 1])
def test_fit_affine_calibrator_small_n_returns_identity(n: int):
    rng = np.random.default_rng(4)
    d = 2
    x = rng.standard_normal((n, d)) if n else np.zeros((0, d))
    y = np.zeros((n, 1), dtype=float)
    model = EpistemicNearestNeighbors(x, y)
    params = ENNParams(
        k_num_neighbors=2,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )
    cal = fit_affine_calibrator(model, params, num_samples=5, rng=rng)
    assert np.allclose(cal.a, 0.0)
    assert np.allclose(cal.b, 1.0)
    assert np.allclose(cal.c, 1.0)


def test_tell_clears_affine_calibrator():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((30, 2))
    y = 1.0 + 2.0 * x[:, :1]
    model = EpistemicNearestNeighbors(x, y)
    fitter = ENNStatefulFitter(k=4, rng=rng)
    fitter.tell(x, y)
    fitter.ask(model, num_fit_candidates=8, num_fit_samples=12, affine_calibrate=True)
    assert fitter.affine_calibrator is not None
    x2 = rng.standard_normal((5, 2))
    y2 = 1.0 + 2.0 * x2[:, :1]
    model.add(x2, y2)
    fitter.tell(x2, y2)
    assert fitter.affine_calibrator is None


def test_fitter_function_draw_applies_calibrator():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((40, 2))
    y = 1.0 + 2.0 * x[:, :1]
    model = EpistemicNearestNeighbors(x, y)
    fitter = ENNStatefulFitter(k=4, rng=rng)
    fitter.tell(x, y)
    params = fitter.ask(
        model, num_fit_candidates=8, num_fit_samples=16, affine_calibrate=True
    )
    q = x[:6]
    raw_draws, _ = model.posterior_function_draw(q, params, function_seeds=[0, 1, 2])
    cal_draws, _ = fitter.posterior_function_draw(
        model, q, params, function_seeds=[0, 1, 2]
    )
    raw_mu = model.posterior(q, params=params).mu
    manual = fitter.affine_calibrator.map_draws(
        np.transpose(raw_draws, (0, 2, 1)), raw_mu
    )
    manual = np.transpose(manual, (0, 2, 1))
    assert np.allclose(cal_draws, manual)
    assert not np.allclose(cal_draws, raw_draws) or (
        abs(float(fitter.affine_calibrator.a[0])) < 1e-9
        and abs(float(fitter.affine_calibrator.b[0]) - 1.0) < 1e-9
        and abs(float(fitter.affine_calibrator.c[0]) - 1.0) < 1e-9
    )


def test_fitter_function_draw_scatter_follows_c_not_b():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((50, 2))
    y = 1.0 + 0.5 * x[:, :1]
    model = EpistemicNearestNeighbors(x, y)
    fitter = ENNStatefulFitter(k=5, rng=rng)
    fitter.tell(x, y)
    params = fitter.ask(
        model, num_fit_candidates=8, num_fit_samples=16, affine_calibrate=False
    )
    fitter.affine_calibrator = AffineCalibrator(
        a=np.array([0.0]),
        b=np.array([3.0]),
        c=np.array([0.5]),
    )
    q = x[:20]
    seeds = list(range(64))
    raw_d, _ = model.posterior_function_draw(q, params, function_seeds=seeds)
    cal_d, _ = fitter.posterior_function_draw(
        model, q, params, function_seeds=seeds
    )
    raw_std = float(np.std(raw_d, axis=-1).mean())
    cal_std = float(np.std(cal_d, axis=-1).mean())
    ratio = cal_std / raw_std
    assert abs(ratio - 0.5) < 0.05
    assert abs(ratio - 3.0) > 1.0
    raw_mu = model.posterior(q, params=params).mu
    mean_cal = cal_d.mean(axis=-1)
    expected_mean = fitter.affine_calibrator.map_mu(raw_mu)
    assert np.allclose(mean_cal, expected_mean, atol=0.05)


def test_fitter_sample_uses_calibrated_posterior():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((35, 2))
    y = 0.5 + 1.5 * x[:, :1]
    model = EpistemicNearestNeighbors(x, y)
    fitter = ENNStatefulFitter(k=4, rng=rng)
    fitter.tell(x, y)
    params = fitter.ask(
        model, num_fit_candidates=8, num_fit_samples=12, affine_calibrate=True
    )
    q = x[:3]
    s1 = fitter.sample(model, q, params, 8, rng=np.random.default_rng(99))
    seed = int(np.random.default_rng(99).integers(0, 2**63 - 1))
    s2 = fitter.posterior(model, q, params).sample(8, seed=seed)
    assert np.allclose(s1, s2)


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

