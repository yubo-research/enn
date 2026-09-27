from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.mbpann import DEFAULT_REBUILD_DRIFT, MBPANNMetric
from enn.turbo.config.enn_index_driver import ENN_INDEX_DRIVER_TO_RUST, ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNSurrogateConfig


def _data(n: int = 400, d: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random((n, d)), rng.random((n, 1))


def _model(tmp_path, driver: ENNIndexDriver = ENNIndexDriver.MBPANN_DISK, n: int = 400):
    x, y = _data(n)
    return x, y, EpistemicNearestNeighbors(x, y, index_driver=driver, work_dir=tmp_path)


def _exact(x: np.ndarray, q: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(((x - q) ** 2) @ w, kind="stable")[:k]


def test_mbpann_maps_to_rust_driver() -> None:
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.MBPANN_DISK] == "mbpann_disk"
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.BPANN_DISK] == "bpann_disk"


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


def test_mbpann_metric_validation(tmp_path) -> None:
    _, _, model = _model(tmp_path)
    with pytest.raises(ValueError, match="rebuild_drift"):
        MBPANNMetric(model, rebuild_drift=-1.0)
    metric = MBPANNMetric(model)
    with pytest.raises(ValueError, match="shape"):
        metric.set_weights(np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        metric.set_weights(np.array([1.0, 0.0, 1.0]))


def test_bpann_disk_rejects_metric_updates(tmp_path) -> None:
    _, _, model = _model(tmp_path, ENNIndexDriver.BPANN_DISK)
    with pytest.raises(ValueError, match="MBPANN_DISK"):
        MBPANNMetric(model)
    with pytest.raises(ValueError, match="MBPANN_DISK"):
        model.rust_backend.set_metric_scale(np.ones(3))


def test_mbpann_rejects_scale_x_and_optimizer_config(tmp_path) -> None:
    x, y = _data(10)
    with pytest.raises(ValueError, match="MBPANN_DISK"):
        EpistemicNearestNeighbors(
            x, y, scale_x=True, index_driver=ENNIndexDriver.MBPANN_DISK, work_dir=tmp_path
        )
    with pytest.raises(ValueError, match="MBPANN_DISK"):
        ENNSurrogateConfig(index_driver=ENNIndexDriver.MBPANN_DISK)


def test_reopen_after_metric_change_uses_identity_metric(tmp_path) -> None:
    x, y, model = _model(tmp_path)
    MBPANNMetric(model).set_weights(np.array([100.0, 1.0, 0.01]))
    model.persist_index_to_disk()
    del model
    reopened = EpistemicNearestNeighbors(
        np.zeros((0, 3)), np.zeros((0, 1)), index_driver=ENNIndexDriver.MBPANN_DISK, work_dir=tmp_path
    )
    q = np.array([[0.5, 0.5, 0.5]])
    np.testing.assert_array_equal(reopened.neighbors(q, 5), _exact(x, q, np.ones(3), 5))
