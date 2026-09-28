from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.mbpann import (
    AUTO_MIN_HELDOUT_GAIN,
    DEFAULT_REBUILD_DRIFT,
    DRIFT_WEIGHT_FLOOR,
    MBPANNMetric,
    auto_uses_learned_metric,
)
from enn._rust import EpistemicNearestNeighbors as RustENN
from enn.turbo.config.enn_index_driver import ENN_INDEX_DRIVER_TO_RUST, ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNSurrogateConfig
from enn.turbo.config.enn_x_scaling import ENNMetricLearning, ENNScaleX
from enn.turbo.config.optimizer_config import OptimizerConfig
from enn.turbo.rust_optimizer_helpers import _config_to_rust_overrides


def _data(n: int = 400, d: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random((n, d)), rng.random((n, 1))


def _model(tmp_path, metric_learning: ENNMetricLearning = ENNMetricLearning.ON, n: int = 400):
    x, y = _data(n)
    model = EpistemicNearestNeighbors(
        x, y, metric_learning=metric_learning, index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tmp_path
    )
    return x, y, model


def _exact(x: np.ndarray, q: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(((x - q) ** 2) @ w, kind="stable")[:k]


def test_scale_x_and_metric_learning_are_separate_enums() -> None:
    assert [d.name for d in ENNIndexDriver] == ["FLAT", "BPANN_DISK"]
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.BPANN_DISK] == "bpann_disk"
    assert [s.name for s in ENNScaleX] == ["OFF", "ON"]
    assert [s.name for s in ENNMetricLearning] == ["OFF", "ON", "AUTO"]


def test_set_weights_rescales_or_rebuilds_and_neighbors_follow_metric(tmp_path) -> None:
    x, y, model = _model(tmp_path)
    metric = MBPANNMetric(model)
    assert metric.set_weights(np.array([4.0, 1.0, 0.1])) is True
    assert metric.set_weights(np.array([4.0, 1.0, 0.12])) is False
    assert (metric.num_rebuilds, metric.num_rescales) == (1, 1)
    np.testing.assert_allclose(metric.weights, [4.0, 1.0, 0.12])
    model.add(*_data(100, seed=1))
    x_all = np.vstack([x, _data(100, seed=1)[0]])
    q = np.array([[0.3, 0.6, 0.9]])
    np.testing.assert_array_equal(model.neighbors(q, 5), _exact(x_all, q, metric.weights, 5))


def test_drift_threshold_extremes(tmp_path) -> None:
    _, _, model = _model(tmp_path)
    always = MBPANNMetric(model, rebuild_drift=0.0)
    assert always.set_weights(np.array([1.0, 1.0, 1.01])) is True
    never = MBPANNMetric(model, rebuild_drift=np.inf)
    assert never.set_weights(np.array([1e4, 1.0, 1e-4])) is False
    assert never.drift(np.array([4.0, 1.0, 1.0])) == pytest.approx(np.log(2.0))
    assert DEFAULT_REBUILD_DRIFT == pytest.approx(np.log(2.0))


def test_drift_ignores_negligible_dimensions(tmp_path) -> None:
    _, _, model = _model(tmp_path)
    metric = MBPANNMetric(model)
    assert metric.set_weights(np.array([1e6, 1e6, 1e-5])) is True
    assert metric.drift(np.array([1e6, 1e6, 1e-3])) == 0.0
    assert metric.set_weights(np.array([1e6, 1e6, 1e-3])) is False
    edge = 1e6 * np.exp(-DRIFT_WEIGHT_FLOOR)
    assert metric.drift(np.array([1e6, 1e6, 4.0 * edge])) == pytest.approx(np.log(2.0))
    assert metric.drift(np.array([4e6, 1e6, 1e-5])) == pytest.approx(np.log(2.0))


def test_mbpann_metric_validation(tmp_path) -> None:
    _, _, model = _model(tmp_path)
    with pytest.raises(ValueError, match="rebuild_drift"):
        MBPANNMetric(model, rebuild_drift=-1.0)
    metric = MBPANNMetric(model)
    with pytest.raises(ValueError, match="shape"):
        metric.set_weights(np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        metric.set_weights(np.array([1.0, 0.0, 1.0]))


def test_bpann_disk_without_metric_learning_rejects_metric_updates(tmp_path) -> None:
    x, _, model = _model(tmp_path, ENNMetricLearning.OFF)
    assert model.metric_learning == ENNMetricLearning.OFF
    assert model.scale_x == ENNScaleX.OFF
    with pytest.raises(ValueError, match="metric_learning=ON or AUTO"):
        MBPANNMetric(model)
    with pytest.raises(ValueError, match="metric_learning=ON or AUTO"):
        model.rust_backend.set_metric_scale(np.ones(3))
    q = np.array([[0.3, 0.6, 0.9]])
    np.testing.assert_array_equal(model.neighbors(q, 5), _exact(x, q, np.ones(3), 5))


def test_scale_x_and_metric_learning_require_matching_driver(tmp_path) -> None:
    x, y = _data(10)
    for mode in (ENNMetricLearning.ON, ENNMetricLearning.AUTO):
        with pytest.raises(ValueError, match=f"metric_learning={mode.name} requires index_driver=BPANN_DISK"):
            EpistemicNearestNeighbors(x, y, metric_learning=mode)
        with pytest.raises(ValueError, match=f"metric_learning={mode.name} requires scale_x=OFF"):
            EpistemicNearestNeighbors(
                x,
                y,
                scale_x=ENNScaleX.ON,
                metric_learning=mode,
                index_driver=ENNIndexDriver.BPANN_DISK,
                work_dir=tmp_path,
            )
    with pytest.raises(ValueError, match="ENNScaleX"):
        EpistemicNearestNeighbors(x, y, scale_x=True)
    with pytest.raises(ValueError, match="ENNMetricLearning"):
        EpistemicNearestNeighbors(x, y, metric_learning=True)
    with pytest.raises(ValueError, match="metric_learning requires"):
        RustENN(x, y, index_driver="exact", metric_learning=True)
    with pytest.raises(ValueError, match="Unknown index_driver"):
        RustENN(x, y, index_driver="mbpann_disk", work_dir=str(tmp_path))


def test_bpann_disk_scale_x_tracks_data_scales_incrementally(tmp_path) -> None:
    rng = np.random.default_rng(3)
    widths = np.array([0.01, 1.0, 100.0])
    x = rng.random((2000, 3)) * widths
    y = rng.random((2000, 1))
    model = EpistemicNearestNeighbors(
        x[:10], y[:10], scale_x=ENNScaleX.ON, index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tmp_path
    )
    assert model.scale_x == ENNScaleX.ON
    for lo in range(10, 2000, 199):
        model.add(x[lo : lo + 199], y[lo : lo + 199])
        model.ensure_index_sync()
    applied = model._x_scale[0]
    np.testing.assert_allclose(applied, x.std(axis=0), rtol=0.011)
    q = rng.random((1, 3)) * widths
    np.testing.assert_array_equal(model.neighbors(q, 5), _exact(x, q, 1.0 / applied**2, 5))


def test_optimizer_config_supports_only_scale_x() -> None:
    with pytest.raises(TypeError, match="metric_learning"):
        ENNSurrogateConfig(metric_learning=ENNMetricLearning.ON)
    assert ENNSurrogateConfig(scale_x=ENNScaleX.ON, index_driver=ENNIndexDriver.BPANN_DISK).scale_x == ENNScaleX.ON
    with pytest.raises(ValueError, match="ENNScaleX"):
        ENNSurrogateConfig(scale_x=True)
    for scale_x, expected in ((ENNScaleX.OFF, None), (ENNScaleX.ON, True)):
        config = OptimizerConfig(surrogate=ENNSurrogateConfig(scale_x=scale_x))
        assert _config_to_rust_overrides(config).get("scale_x") is expected


def test_reopen_after_metric_change_uses_identity_metric(tmp_path) -> None:
    x, y, model = _model(tmp_path)
    MBPANNMetric(model).set_weights(np.array([100.0, 1.0, 0.01]))
    model.persist_index_to_disk()
    del model
    reopened = EpistemicNearestNeighbors(
        np.zeros((0, 3)),
        np.zeros((0, 1)),
        metric_learning=ENNMetricLearning.ON,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=tmp_path,
    )
    q = np.array([[0.5, 0.5, 0.5]])
    np.testing.assert_array_equal(reopened.neighbors(q, 5), _exact(x, q, np.ones(3), 5))


def test_auto_rule_needs_positive_finite_heldout_gain() -> None:
    assert AUTO_MIN_HELDOUT_GAIN == 0.0
    assert auto_uses_learned_metric(0.01) is True
    for gain in (0.0, -0.5, -np.inf, np.nan, np.inf):
        assert auto_uses_learned_metric(gain) is False


def test_auto_applies_validated_weights_else_identity_like_none(tmp_path) -> None:
    x, y, model = _model(tmp_path, ENNMetricLearning.AUTO)
    none_dir = tmp_path / "none"
    none = EpistemicNearestNeighbors(
        x, y, metric_learning=ENNMetricLearning.OFF, index_driver=ENNIndexDriver.BPANN_DISK, work_dir=none_dir
    )
    metric = MBPANNMetric(model)
    assert metric.metric_learning == ENNMetricLearning.AUTO
    q = np.array([[0.3, 0.6, 0.9]])
    w = np.array([4.0, 1.0, 0.1])
    assert metric.set_weights_if_validated(w, heldout_gain=-0.1) is False
    assert (metric.num_rebuilds, metric.num_rescales) == (0, 0)
    np.testing.assert_array_equal(model.neighbors(q, 5), none.neighbors(q, 5))
    assert metric.set_weights_if_validated(w, heldout_gain=0.2) is True
    np.testing.assert_allclose(metric.weights, w)
    np.testing.assert_array_equal(model.neighbors(q, 5), _exact(x, q, w, 5))
    assert metric.set_weights_if_validated(w, heldout_gain=-0.2) is False
    np.testing.assert_array_equal(metric.weights, np.ones(3))
    np.testing.assert_array_equal(model.neighbors(q, 5), none.neighbors(q, 5))


def test_set_weights_if_validated_requires_auto(tmp_path) -> None:
    _, _, model = _model(tmp_path)
    with pytest.raises(ValueError, match="requires metric_learning=AUTO"):
        MBPANNMetric(model).set_weights_if_validated(np.ones(3), 1.0)
