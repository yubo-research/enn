from __future__ import annotations

import numpy as np
import pytest

from enn.enn import metric_stream as ms


def _two_of_six(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((n, 6))
    return x, np.sin(4 * x[:, 0]) + x[:, 1] + 0.05 * rng.standard_normal(n)


def test_reservoir_fills_then_keeps_capacity() -> None:
    res = ms.Reservoir(5, 2, np.random.default_rng(0))
    res.add(np.arange(6).reshape(3, 2), np.arange(3))
    assert len(res) == 3 and res.num_seen == 3
    np.testing.assert_array_equal(res.y, [0, 1, 2])
    res.add(np.zeros((10, 2)), np.full(10, -1.0))
    assert len(res) == 5 and res.capacity == 5 and res.num_seen == 13
    assert res.x.shape == (5, 2)


def test_reservoir_is_uniform_over_stream() -> None:
    counts = np.zeros(40)
    for seed in range(400):
        res = ms.Reservoir(10, 1, np.random.default_rng(seed))
        for lo in range(0, 40, 7):
            ids = np.arange(lo, min(40, lo + 7), dtype=float)
            res.add(ids[:, None], ids)
        counts[res.y.astype(int)] += 1
    assert np.all(np.abs(counts / 400 - 0.25) < 0.08)


def test_reservoir_version_counts_stores() -> None:
    res = ms.Reservoir(3, 1, np.random.default_rng(0))
    res.add(np.zeros((2, 1)), np.zeros(2))
    assert res.version == 1
    res.add(np.zeros((0, 1)), np.zeros(0))
    assert res.version == 1
    res.add(np.ones((1, 1)), np.ones(1))
    assert res.version == 2


def test_learner_cache_follows_reservoir() -> None:
    x, y = _two_of_six(800)
    lrn = ms.PerturbationMetricLearner(6, np.random.default_rng(8), capacity=50)
    for lo in range(0, 800, 1):
        lrn.update(x[lo : lo + 1], y[lo : lo + 1])
        if lo % 200 == 199:
            mean, std, zr, zr_sq, yz, _, _ = lrn._standardized_reservoir()
            xr, yr = lrn.reservoir.x, lrn.reservoir.y
            np.testing.assert_allclose(zr, (xr - xr.mean(0)) / xr.std(0))
            np.testing.assert_allclose(zr_sq, zr * zr)
            np.testing.assert_allclose(yz, (yr - yr.mean()) / yr.std())


def test_reservoir_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="capacity"):
        ms.Reservoir(0, 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="rows"):
        ms.Reservoir(3, 2, np.random.default_rng(0)).add(np.zeros((2, 2)), np.zeros(3))


def test_indices_find_relevant_inputs() -> None:
    x, y = _two_of_six(2000)
    for s in (ms.sobol_index(x, y), ms.correlation_index(x, y)):
        assert s[:2].min() > 5 * s[2:].max()
    assert ms.sobol_index(x[:3], y[:3]).sum() == 0
    assert ms.correlation_index(x[:2], y[:2]).sum() == 0
    assert np.isinf(ms.null_sd("sobol", 2))


def test_sobol_index_sees_nonmonotone_dependence() -> None:
    rng = np.random.default_rng(1)
    x = rng.random((3000, 2))
    y = np.sin(6 * np.pi * x[:, 0])
    assert ms.correlation_index(x, y)[0] < 0.2
    assert ms.sobol_index(x, y)[0] > 0.9


def test_dependence_weights_scale_and_fallback() -> None:
    x, y = _two_of_six(2000)
    x = x * np.array([1.0, 10.0, 1, 1, 1, 1])
    w = ms.dependence_weights(x, y)
    assert w[2:].max() < 1e-3 * w[:2].min()
    np.testing.assert_allclose(ms.dependence_weights(x, np.ones(2000)), 1.0 / x.var(axis=0))
    with pytest.raises(ValueError, match="index"):
        ms.dependence_weights(x, y, "nope")


def test_validated_weights_choose_dependence_or_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = _two_of_six(400)
    np.testing.assert_allclose(ms.validated_dependence_weights(x, y, 5), ms.dependence_weights(x, y))
    np.testing.assert_allclose(ms.validated_dependence_weights(x[:50], y[:50], 5), 1.0 / x[:50].var(axis=0))
    wrong = np.array([1e-6, 1e-6, 1.0, 1.0, 1e-6, 1e-6])
    monkeypatch.setattr(ms, "dependence_weights", lambda *a, **kw: wrong)
    np.testing.assert_allclose(ms.validated_dependence_weights(x, y, 5), 1.0 / x.var(axis=0))


def test_loo_loglik_prefers_right_metric() -> None:
    x, y = _two_of_six(400)
    good = np.array([1.0, 1, 1e-6, 1e-6, 1e-6, 1e-6])
    assert ms.loo_loglik(x, y, good, 5) > ms.loo_loglik(x, y, np.ones(6), 5)


def test_perturbed_loglik_matches_exact_knn() -> None:
    rng = np.random.default_rng(5)
    xq, xr = rng.random((4, 3)), rng.random((30, 3))
    yq, yr = rng.random(4), rng.random(30)
    theta = np.tile([0.0, 1.0, -1.0, np.log(0.1)], (4, 1))
    (ll,) = ms.perturbed_loglik(xq, yq, xr, yr, [theta], 3)
    a = np.exp(theta[0, :-1])
    d2 = ((xq[:, None, :] - xr[None]) ** 2) @ a
    nbr = np.argsort(d2, axis=1)[:, :3]
    w = 1.0 / (ms.EPS + np.take_along_axis(d2, nbr, axis=1) + 0.1)
    mu = (w * yr[nbr]).sum(1) / w.sum(1)
    var = 1.0 / w.sum(1) + 0.1
    np.testing.assert_allclose(ll, -0.5 * np.log(2 * np.pi * var) - 0.5 * (yq - mu) ** 2 / var)


def test_learner_moves_toward_relevant_inputs() -> None:
    x, y = _two_of_six(6000)
    lrn = ms.PerturbationMetricLearner(6, np.random.default_rng(6), capacity=300)
    for lo in range(0, 6000, 100):
        lrn.update(x[lo : lo + 100], y[lo : lo + 100])
    t = lrn.theta[:-1]
    assert t[:2].min() > t[2:].max()
    assert lrn.num_updates == 6000 - 100 and len(lrn.reservoir) == 300
    np.testing.assert_allclose(lrn.weights, np.exp(t) / lrn.reservoir.x.var(axis=0))


def test_learner_starts_from_validated_dependence_weights() -> None:
    x, y = _two_of_six(400)
    lrn = ms.PerturbationMetricLearner(6, np.random.default_rng(7), init_rows=200)
    lrn.update(x[:200], y[:200])
    assert np.all(lrn.theta[:-1] == 0)
    lrn.update(x[200:201], y[200:201])
    assert lrn.theta[:2].min() > lrn.theta[2:-1].max() + 5


def test_learner_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="perturbation"):
        ms.PerturbationMetricLearner(3, np.random.default_rng(0), step=0.0)
