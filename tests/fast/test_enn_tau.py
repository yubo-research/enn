"""QA tests for the prior precision ``tau``, which acts on centered, scaled data.

mu = m + sum(w (y - m)) / (tau + sum(w)),  se = y_scale / sqrt(tau + sum(w)),
with m the training mean. With S = sum(w) = (y_scale / se_0)^2 recovered from the
tau=0 posterior:
  mu_tau - m = (mu_0 - m) * S / (S + tau),  1/se_tau^2 = (S + tau) / y_scale^2.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from enn import ENNStatefulFitter, EpistemicNearestNeighbors
from enn.enn.enn_fitter import subsample_loglik
from enn.enn.enn_params import ENNParams, PosteriorFlags
from evals.harness.work_dirs import fresh_work_dir


def _data(n=30, d=3, seed=0, offset=0.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, d))
    y = np.sin(3 * x[:, :1]) + 0.3 * rng.normal(size=(n, 1)) + offset
    return x, y


def _model(x, y):
    return EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())


def _params(tau=0.0, k=5, ale=0.1):
    return ENNParams(
        k_num_neighbors=k,
        epistemic_variance_scale=0.7,
        aleatoric_variance_scale=ale,
        tau=tau,
    )


@pytest.mark.parametrize("tau", [0.01, 0.3, 1.0, 10.0, 300.0])
def test_tau_matches_closed_form(tau):
    x, y = _data(offset=2.0)
    q = np.random.default_rng(1).uniform(size=(7, 3))
    model = _model(x, y)
    p0 = model.posterior(q, params=_params(0.0))
    pt = model.posterior(q, params=_params(tau))
    ys = float(np.std(y))
    s = (ys / p0.se) ** 2
    m = float(np.mean(y))
    np.testing.assert_allclose(pt.mu - m, (p0.mu - m) * s / (s + tau), rtol=1e-9)
    np.testing.assert_allclose(1 / pt.se**2, (s + tau) / ys**2, rtol=1e-9)


def test_tau_zero_is_default_and_bit_identical():
    x, y = _data()
    q = x[:5] + 0.01
    model = _model(x, y)
    base = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=0.7, aleatoric_variance_scale=0.1
    )
    assert base.tau == 0.0
    a = model.posterior(q, params=base)
    b = model.posterior(q, params=_params(0.0))
    np.testing.assert_array_equal(a.mu, b.mu)
    np.testing.assert_array_equal(a.se, b.se)


def test_tau_limits_and_monotonicity():
    x, y = _data(offset=3.0)
    q = np.array([[0.5, 0.5, 0.5]])
    model = _model(x, y)
    mus = [
        abs(model.posterior(q, params=_params(t)).mu.item() - float(np.mean(y)))
        for t in (0, 0.1, 1, 10, 1e3)
    ]
    ses = [
        model.posterior(q, params=_params(t)).se.item() for t in (0, 0.1, 1, 10, 1e3)
    ]
    assert all(a > b for a, b in itertools.pairwise(mus))
    assert all(a > b for a, b in itertools.pairwise(ses))
    assert mus[-1] < 0.05 * mus[0]


def test_tau_far_from_data_posterior_stays_finite_and_shrinks_to_training_mean():
    x, y = _data()
    q = np.array([[1e3, 1e3, 1e3]])
    model = _model(x, y)
    post = model.posterior(q, params=_params(5.0))
    assert np.isfinite(post.mu).all() and np.isfinite(post.se).all()
    assert abs(post.mu.item() - float(np.mean(y))) < 1e-3


def test_tau_observation_noise_and_decomposition():
    x, y = _data()
    q = x[:4] + 0.02
    model = _model(x, y)
    post = model.posterior(
        q, params=_params(2.0), flags=PosteriorFlags(observation_noise=True)
    )
    assert np.all(post.se_ale > 0)
    np.testing.assert_allclose(post.se**2, post.se_epi**2 + post.se_ale**2, rtol=1e-9)


def test_tau_sampling_marginal_matches_se():
    x, y = _data()
    q = np.array([[0.4, 0.4, 0.4]])
    model = _model(x, y)
    params = _params(3.0, ale=0.0)
    post = model.posterior(q, params=params)
    draws, _ = model.posterior_function_draw(
        q, params=params, function_seeds=list(range(4000))
    )
    d = np.asarray(draws).reshape(4000, -1)[:, 0]
    assert abs(d.mean() - post.mu.item()) < 0.06 * post.se.item()
    assert abs(d.std() / post.se.item() - 1.0) < 0.08


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_tau_validation(bad):
    with pytest.raises(ValueError):
        _params(bad)


def test_tau_batch_matches_individual():
    x, y = _data()
    q = x[:6] * 0.9
    model = _model(x, y)
    paramss = [_params(t) for t in (0.0, 0.5, 20.0)]
    batch = model.batch_posterior(q, paramss=paramss)
    for i, p in enumerate(paramss):
        one = model.posterior(q, params=p)
        np.testing.assert_allclose(np.asarray(batch.mu)[i], one.mu, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(batch.se)[i], one.se, rtol=1e-12)


def test_tau_subsample_loglik_prefers_prior_for_zero_mean_noise():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(40, 2))
    y = 0.5 * rng.normal(size=(40, 1))
    model = _model(x, y)
    lls = subsample_loglik(
        model, x, y, paramss=[_params(0.0, k=3), _params(1.0, k=3)], P=40, seed=0
    )
    assert np.isfinite(lls).all()
    assert lls[0] != lls[1]


def test_fitter_tau_off_by_default_and_fit_when_enabled():
    x, y = _data(n=60)
    model = _model(x, y)
    off = ENNStatefulFitter(5, seed=0).ask(model, num_fit_candidates=20)
    assert off.tau == 0.0
    on = ENNStatefulFitter(5, seed=0, infer_prior_precision=True).ask(
        model, num_fit_candidates=20
    )
    assert on.tau > 0.0 and np.isfinite(on.tau)


def test_fitter_tau_with_affine_calibration_runs():
    x, y = _data(n=60)
    model = _model(x, y)
    p = ENNStatefulFitter(5, seed=1, infer_prior_precision=True).ask(
        model, num_fit_candidates=10, affine_calibrate=True
    )
    assert p.calibrator is not None and p.tau > 0.0


@pytest.mark.parametrize("tau", [0.0, 0.3, 20.0])
def test_tau_posterior_is_equivariant_to_shift_and_scale_of_y(tau):
    x, y = _data()
    q = np.random.default_rng(2).uniform(size=(6, 3)) * 3.0
    a = _model(x, y).posterior(q, params=_params(tau))
    b = _model(x, 4.0 * y + 50.0).posterior(q, params=_params(tau))
    np.testing.assert_allclose(b.mu, 4.0 * a.mu + 50.0, rtol=1e-9)
    np.testing.assert_allclose(b.se, 4.0 * a.se, rtol=1e-9)
