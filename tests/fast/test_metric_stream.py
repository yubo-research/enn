from __future__ import annotations

import numpy as np
import pytest

from enn.enn import metric_stream as ms


def _two_of_six(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((n, 6))
    return x, np.sin(4 * x[:, 0]) + x[:, 1] + 0.05 * rng.standard_normal(n)


def test_reservoir_fills_then_keeps_capacity() -> None:
    res = ms.Reservoir(5, 2, 0)
    res.add(np.arange(6).reshape(3, 2), np.arange(3))
    assert len(res) == 3 and res.num_seen == 3
    np.testing.assert_array_equal(res.y, [[0], [1], [2]])
    res.add(np.zeros((10, 2)), np.full(10, -1.0))
    assert len(res) == 5 and res.capacity == 5 and res.num_seen == 13
    assert res.x.shape == (5, 2)


def test_reservoir_keeps_output_columns() -> None:
    res = ms.Reservoir(4, 1, 0, num_outputs=2)
    res.add(np.zeros((3, 1)), np.arange(6.0).reshape(3, 2))
    np.testing.assert_array_equal(res.y, [[0, 1], [2, 3], [4, 5]])
    with pytest.raises(ValueError, match="columns"):
        res.add(np.zeros((1, 1)), np.zeros(1))


def test_reservoir_is_uniform_over_stream() -> None:
    counts = np.zeros(40)
    for seed in range(400):
        res = ms.Reservoir(10, 1, seed)
        for lo in range(0, 40, 7):
            ids = np.arange(lo, min(40, lo + 7), dtype=float)
            res.add(ids[:, None], ids)
        counts[res.y[:, 0].astype(int)] += 1
    assert np.all(np.abs(counts / 400 - 0.25) < 0.08)


def test_reservoir_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="capacity"):
        ms.Reservoir(0, 2, 0)
    with pytest.raises(ValueError, match="rows"):
        ms.Reservoir(3, 2, 0).add(np.zeros((2, 2)), np.zeros(3))


def test_sobol_index_finds_relevant_inputs() -> None:
    x, y = _two_of_six(2000)
    s = ms.sobol_index(x, y)
    assert s[:2].min() > 5 * s[2:].max()
    assert ms.sobol_index(x[:3], y[:3]).sum() == 0
    assert np.isinf(ms.null_sd(2))


def test_sobol_index_sees_nonmonotone_dependence() -> None:
    rng = np.random.default_rng(1)
    x = rng.random((3000, 2))
    assert ms.sobol_index(x, np.sin(6 * np.pi * x[:, 0]))[0] > 0.9


def test_dependence_weights_scale_and_fallback() -> None:
    x, y = _two_of_six(2000)
    x = x * np.array([1.0, 10.0, 1, 1, 1, 1])
    w = ms.dependence_weights(x, y)
    assert w[2:].max() < 1e-3 * w[:2].min()
    np.testing.assert_allclose(
        ms.dependence_weights(x, np.ones(2000)), 1.0 / x.var(axis=0)
    )


def test_dependence_weights_average_output_columns() -> None:
    x, y = _two_of_six(2000)
    both = ms.dependence_weights(x, np.column_stack([y, x[:, 2]]))
    assert both[:3].min() > 1e3 * both[3:].max()


def test_loo_loglik_prefers_right_metric() -> None:
    x, y = _two_of_six(400)
    good = np.array([1.0, 1, 1e-6, 1e-6, 1e-6, 1e-6])
    assert ms.loo_loglik(x, y, good, 5) > ms.loo_loglik(x, y, np.ones(6), 5)
    y2 = np.column_stack([y, y])
    assert ms.loo_loglik(x, y2, good, 5) == pytest.approx(ms.loo_loglik(x, y, good, 5))


def test_bins_floor_and_k_change_the_rust_result() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((200, 3))
    y = x[:, 0] + 0.01 * rng.standard_normal(200)
    assert not np.array_equal(ms.sobol_index(x, y), ms.sobol_index(x, y, num_bins=2))
    assert not np.allclose(
        ms.dependence_weights(x, y), ms.dependence_weights(x, y, floor=0.5)
    )
    _, gain_1 = ms.auto_weights(x, y, 1)
    _, gain_10 = ms.auto_weights(x, y, 10)
    assert gain_1 != gain_10


def test_auto_weights_gain_sign_tracks_signal() -> None:
    x, y = _two_of_six(400)
    w, gain = ms.auto_weights(x, y, 5)
    np.testing.assert_allclose(w, ms.dependence_weights(x, y))
    assert gain > 0.3
    rng = np.random.default_rng(2)
    w_sphere, gain_sphere = ms.auto_weights(
        x, ((x - 0.5) ** 2).sum(1) + 0.01 * rng.standard_normal(400), 5
    )
    assert gain_sphere < 0.05
    w_small, gain_small = ms.auto_weights(x[:50], y[:50], 5)
    assert gain_small == float("-inf")
    np.testing.assert_array_equal(w_small, np.ones(6))
