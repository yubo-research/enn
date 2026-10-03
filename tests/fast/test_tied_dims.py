from __future__ import annotations

import numpy as np
import pytest

from enn.enn import metric_stream as ms
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNStorage
from enn.turbo.config.enn_x_scaling import ENNMetricLearning, ENNScaleX, validate_tied_dims


def _mixed(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Columns: 0 continuous (wide, relevant), 1 continuous noise, 2-4 one-hot category (relevant)."""
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 3, n)
    x = np.column_stack([10 * rng.random(n), rng.random(n), np.eye(3)[cat]])
    y = np.sin(0.4 * x[:, 0]) + np.array([0.0, 1.0, -1.0])[cat] + 0.05 * rng.standard_normal(n)
    return x, y


def test_group_sobol_index_sees_category_effect() -> None:
    x, y = _mixed(1000)
    assert ms.group_sobol_index(x[:, 2:5], y) > 0.5
    assert ms.group_sobol_index(x[:, 2:5], np.random.default_rng(1).random(1000)) < 0.02
    assert ms.group_sobol_index(x[:5, 2:5], y[:5]) == 0.0
    assert ms.group_sobol_index(np.ones((50, 2)), y[:50]) == 0.0
    assert ms.null_sd(1000, 3) == pytest.approx(2.0 / 1000)


def test_dependence_weights_share_one_unscaled_weight_per_group() -> None:
    x, y = _mixed(1000)
    tied = ((2, 3, 4),)
    w = ms.dependence_weights(x, y, tied=tied)
    assert w[2] == w[3] == w[4]
    s = np.array([ms.sobol_index(x, y)[0], ms.group_sobol_index(x[:, 2:5], y)])
    np.testing.assert_allclose(w[[0, 2]] * [x[:, 0].var(), 1.0], 3 * s / s.sum(), rtol=1e-3)
    assert w[1] < 1e-3 * w[2]
    fallback = ms.dependence_weights(x, np.ones(1000), tied=tied)
    np.testing.assert_allclose(fallback, [1 / x[:, 0].var(), 1 / x[:, 1].var(), 1, 1, 1])
    np.testing.assert_array_equal(ms.dependence_weights(x, y), ms.dependence_weights(x, y, tied=()))
    w_auto, gain = ms.auto_weights(x, y, 10, tied=tied)
    np.testing.assert_array_equal(w_auto, w)
    assert gain > 0


def test_validate_tied_dims() -> None:
    assert validate_tied_dims(None, 4) == ()
    assert validate_tied_dims([np.array([2, 3]), [0]], 4) == ((2, 3), (0,))
    with pytest.raises(TypeError):
        validate_tied_dims([[1.5]], 4)
    for bad, match in [([[]], "non-empty"), ([[1, 4]], r"\[0, 4\)"), ([[-1]], r"\[0, 4\)"), ([[0, 1], [1, 2]], "disjoint")]:
        with pytest.raises(ValueError, match=match):
            validate_tied_dims(bad, 4)


@pytest.mark.parametrize("driver", [ENNIndexDriver.FLAT, ENNIndexDriver.BPANN_DISK])
def test_scale_x_leaves_tied_dims_unscaled(tmp_path, driver) -> None:
    x, y = _mixed(600)
    kw = (
        {"work_dir": tmp_path, "enn_storage": ENNStorage.DISK}
        if driver == ENNIndexDriver.BPANN_DISK
        else {}
    )
    model = EpistemicNearestNeighbors(
        x[:300], y[:300, None], tied_dims=[[2, 3, 4]], scale_x=ENNScaleX.ON, index_driver=driver, **kw
    )
    assert model.tied_dims == ((2, 3, 4),)
    model.add(x[300:], y[300:, None])
    scale = model._x_scale[0]
    np.testing.assert_array_equal(scale[2:], 1.0)
    np.testing.assert_allclose(scale[:2], x[:, :2].std(axis=0), rtol=0.011)
    q = x[:4] + 0.01
    for i in range(len(q)):
        d2 = (((x - q[i]) / scale) ** 2).sum(axis=1)
        np.testing.assert_array_equal(model.neighbors(q[i], 5), np.argsort(d2, kind="stable")[:5])
    with pytest.raises(ValueError, match="disjoint"):
        EpistemicNearestNeighbors(x, y[:, None], tied_dims=[[2, 3], [3, 4]])


def test_auto_metric_ties_group_weights(tmp_path) -> None:
    x, y = _mixed(700)
    model = EpistemicNearestNeighbors(
        x[:50],
        y[:50, None],
        tied_dims=[[2, 3, 4]],
        metric_learning=ENNMetricLearning.AUTO,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=tmp_path,
        enn_storage=ENNStorage.DISK,
    )
    for lo in range(50, 700, 50):
        model.add(x[lo : lo + 50], y[lo : lo + 50, None])
    metric = model.metric
    assert metric.uses_learned_metric
    w = metric.weights
    assert w[2] == w[3] == w[4] and w[2] > 1e3 * w[1]
    q = x[:3] + 0.01
    for i in range(len(q)):
        d2 = ((x - q[i]) ** 2) @ w
        np.testing.assert_array_equal(model.neighbors(q[i], 5), np.argsort(d2, kind="stable")[:5])
