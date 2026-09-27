from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.mbpann import (
    DEFAULT_REBUILD_DRIFT,
    DRIFT_WEIGHT_FLOOR,
    MBPANNMetric,
)
from enn._rust import EpistemicNearestNeighbors as RustENN
from enn.turbo.config.enn_index_driver import ENN_INDEX_DRIVER_TO_RUST, ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNSurrogateConfig
from enn.turbo.config.enn_x_scaling import ENNXScaling
from enn.turbo.config.optimizer_config import OptimizerConfig
from enn.turbo.rust_optimizer_helpers import _config_to_rust_overrides


def _data(n: int = 400, d: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random((n, d)), rng.random((n, 1))


def _model(tmp_path, x_scaling: ENNXScaling = ENNXScaling.METRIC_LEARNING, n: int = 400):
    x, y = _data(n)
    model = EpistemicNearestNeighbors(
        x, y, x_scaling=x_scaling, index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tmp_path
    )
    return x, y, model


def _exact(x: np.ndarray, q: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(((x - q) ** 2) @ w, kind="stable")[:k]


def test_metric_learning_is_a_bpann_disk_x_scaling() -> None:
    assert [d.name for d in ENNIndexDriver] == ["FLAT", "BPANN_DISK"]
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.BPANN_DISK] == "bpann_disk"
    assert [s.name for s in ENNXScaling] == ["NONE", "SCALE_X", "METRIC_LEARNING"]


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
    x, _, model = _model(tmp_path, ENNXScaling.NONE)
    assert model.x_scaling == ENNXScaling.NONE
    with pytest.raises(ValueError, match="METRIC_LEARNING"):
        MBPANNMetric(model)
    with pytest.raises(ValueError, match="METRIC_LEARNING"):
        model.rust_backend.set_metric_scale(np.ones(3))
    q = np.array([[0.3, 0.6, 0.9]])
    np.testing.assert_array_equal(model.neighbors(q, 5), _exact(x, q, np.ones(3), 5))


def test_x_scaling_requires_matching_driver(tmp_path) -> None:
    x, y = _data(10)
    with pytest.raises(ValueError, match="SCALE_X requires index_driver=FLAT"):
        EpistemicNearestNeighbors(
            x, y, x_scaling=ENNXScaling.SCALE_X, index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tmp_path
        )
    with pytest.raises(ValueError, match="METRIC_LEARNING requires index_driver=BPANN_DISK"):
        EpistemicNearestNeighbors(x, y, x_scaling=ENNXScaling.METRIC_LEARNING)
    with pytest.raises(ValueError, match="ENNXScaling"):
        EpistemicNearestNeighbors(x, y, x_scaling=True)
    with pytest.raises(ValueError, match="metric_learning requires"):
        RustENN(x, y, index_driver="exact", metric_learning=True)
    with pytest.raises(ValueError, match="Unknown index_driver"):
        RustENN(x, y, index_driver="mbpann_disk", work_dir=str(tmp_path))


def test_optimizer_config_rejects_metric_learning() -> None:
    with pytest.raises(ValueError, match="METRIC_LEARNING is an ENN-model mode"):
        ENNSurrogateConfig(x_scaling=ENNXScaling.METRIC_LEARNING, index_driver=ENNIndexDriver.BPANN_DISK)
    with pytest.raises(ValueError, match="SCALE_X requires index_driver=FLAT"):
        ENNSurrogateConfig(x_scaling=ENNXScaling.SCALE_X, index_driver=ENNIndexDriver.BPANN_DISK)
    for x_scaling, expected in ((ENNXScaling.NONE, None), (ENNXScaling.SCALE_X, True)):
        config = OptimizerConfig(surrogate=ENNSurrogateConfig(x_scaling=x_scaling))
        assert _config_to_rust_overrides(config).get("scale_x") is expected


def test_reopen_after_metric_change_uses_identity_metric(tmp_path) -> None:
    x, y, model = _model(tmp_path)
    MBPANNMetric(model).set_weights(np.array([100.0, 1.0, 0.01]))
    model.persist_index_to_disk()
    del model
    reopened = EpistemicNearestNeighbors(
        np.zeros((0, 3)),
        np.zeros((0, 1)),
        x_scaling=ENNXScaling.METRIC_LEARNING,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=tmp_path,
    )
    q = np.array([[0.5, 0.5, 0.5]])
    np.testing.assert_array_equal(reopened.neighbors(q, 5), _exact(x, q, np.ones(3), 5))
