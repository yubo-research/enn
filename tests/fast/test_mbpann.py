from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.mbpann import (
    AUTO_MIN_HELDOUT_GAIN,
    AUTO_REFIT_GROWTH,
    AUTO_RESERVOIR_CAPACITY,
    DEFAULT_REBUILD_DRIFT,
    DRIFT_WEIGHT_FLOOR,
    MBPANNMetric,
    auto_uses_learned_metric,
)
from enn._rust import EpistemicNearestNeighbors as RustENN
from enn.turbo.config.enn_index_driver import ENN_INDEX_DRIVER_TO_RUST, ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNStorage, ENNSurrogateConfig
from enn.turbo.config.enn_x_scaling import ENNMetricLearning, ENNScaleX
from enn.turbo.config.optimizer_config import OptimizerConfig
from enn.turbo.rust_optimizer_helpers import _config_to_rust_overrides


def _data(n: int = 400, d: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random((n, d)), rng.random((n, 1))


def _model(
    tmp_path, metric_learning: ENNMetricLearning = ENNMetricLearning.AUTO, n: int = 50
):
    x, y = _data(n)
    model = EpistemicNearestNeighbors(
        x,
        y,
        metric_learning=metric_learning,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=tmp_path,
        enn_storage=ENNStorage.DISK,
    )
    return x, y, model


def _exact(x: np.ndarray, q: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(((x - q) ** 2) @ w, kind="stable")[:k]


def test_scale_x_and_metric_learning_are_separate_enums() -> None:
    assert [d.name for d in ENNIndexDriver] == ["FLAT", "BPANN_DISK"]
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.BPANN_DISK] == "BPANN_DISK"
    assert ENN_INDEX_DRIVER_TO_RUST[ENNIndexDriver.FLAT] == "FLAT"
    assert [s.name for s in ENNScaleX] == ["OFF", "ON"]
    assert [s.name for s in ENNMetricLearning] == ["NONE", "AUTO"]


def test_set_weights_rescales_or_rebuilds_and_neighbors_follow_metric(tmp_path) -> None:
    x, y, model = _model(tmp_path)
    metric = model.metric
    assert metric.set_weights(np.array([4.0, 1.0, 0.1])) is True
    assert metric.set_weights(np.array([4.0, 1.0, 0.12])) is False
    assert (metric.num_rebuilds, metric.num_rescales) == (1, 1)
    np.testing.assert_allclose(metric.weights, [4.0, 1.0, 0.12])
    model.add(*_data(40, seed=1))
    x_all = np.vstack([x, _data(40, seed=1)[0]])
    q = np.array([[0.3, 0.6, 0.9]])
    np.testing.assert_array_equal(
        model.neighbors(q, 5), _exact(x_all, q, metric.weights, 5)
    )


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
    metric = model.metric
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
    with pytest.raises(ValueError, match="refit_growth"):
        MBPANNMetric(model, refit_growth=1.0)
    metric = model.metric
    with pytest.raises(ValueError, match="dimension"):
        metric.set_weights(np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        metric.set_weights(np.array([1.0, 0.0, 1.0]))


def test_bpann_disk_without_metric_learning_rejects_metric_updates(tmp_path) -> None:
    x, _, model = _model(tmp_path, ENNMetricLearning.NONE)
    assert model.metric_learning == ENNMetricLearning.NONE
    assert model.scale_x == ENNScaleX.OFF
    assert model.metric is None
    with pytest.raises(ValueError, match="metric_learning=AUTO"):
        MBPANNMetric(model)
    with pytest.raises(ValueError, match="metric_learning=AUTO"):
        model.rust_backend.set_metric_scale(np.ones(3))
    q = np.array([[0.3, 0.6, 0.9]])
    np.testing.assert_array_equal(model.neighbors(q, 5), _exact(x, q, np.ones(3), 5))


def test_scale_x_and_metric_learning_require_matching_driver(tmp_path) -> None:
    x, y = _data(10)
    for mode in (ENNMetricLearning.AUTO,):
        with pytest.raises(
            ValueError,
            match="BpAnnDisk",
        ):
            EpistemicNearestNeighbors(x, y, metric_learning=mode)
        with pytest.raises(
            ValueError, match="scale_x=false"
        ):
            EpistemicNearestNeighbors(
                x,
                y,
                scale_x=ENNScaleX.ON,
                metric_learning=mode,
                index_driver=ENNIndexDriver.BPANN_DISK,
                work_dir=tmp_path,
                enn_storage=ENNStorage.DISK,
            )
    with pytest.raises(ValueError, match="ENNScaleX"):
        EpistemicNearestNeighbors(x, y, scale_x=True)
    with pytest.raises(ValueError, match="ENNMetricLearning"):
        EpistemicNearestNeighbors(x, y, metric_learning=True)
    with pytest.raises(ValueError, match="Unknown index_driver"):
        RustENN(x, y, index_driver="exact", metric_learning="auto")
    with pytest.raises(ValueError, match="Unknown index_driver"):
        RustENN(x, y, index_driver="mbpann_disk", work_dir=str(tmp_path))


def test_bpann_disk_scale_x_tracks_data_scales_incrementally(tmp_path) -> None:
    rng = np.random.default_rng(3)
    widths = np.array([0.01, 1.0, 100.0])
    x = rng.random((2000, 3)) * widths
    y = rng.random((2000, 1))
    model = EpistemicNearestNeighbors(
        x[:10],
        y[:10],
        scale_x=ENNScaleX.ON,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=tmp_path,
        enn_storage=ENNStorage.DISK,
    )
    assert model.scale_x == ENNScaleX.ON
    for lo in range(10, 2000, 199):
        model.add(x[lo : lo + 199], y[lo : lo + 199])
        model.ensure_index_sync()
    applied = model._x_scale[0]
    np.testing.assert_allclose(applied, x.std(axis=0), rtol=0.011)
    q = rng.random((1, 3)) * widths
    np.testing.assert_array_equal(
        model.neighbors(q, 5), _exact(x, q, 1.0 / applied**2, 5)
    )


def test_optimizer_config_forwards_scale_x_and_metric_learning() -> None:
    config = ENNSurrogateConfig(
        metric_learning=ENNMetricLearning.AUTO,
        index_driver=ENNIndexDriver.BPANN_DISK,
        enn_storage=ENNStorage.DISK,
        work_dir="/tmp/enn_auto",
        scale_x=ENNScaleX.OFF,
    )
    assert config.metric_learning == ENNMetricLearning.AUTO
    assert (
        ENNSurrogateConfig(
            scale_x=ENNScaleX.ON, index_driver=ENNIndexDriver.BPANN_DISK
        ).scale_x
        == ENNScaleX.ON
    )
    with pytest.raises(ValueError, match="ENNScaleX"):
        ENNSurrogateConfig(scale_x=True)
    for scale_x, expected in ((ENNScaleX.OFF, None), (ENNScaleX.ON, True)):
        config = OptimizerConfig(surrogate=ENNSurrogateConfig(scale_x=scale_x))
        assert _config_to_rust_overrides(config).get("scale_x") is expected


def test_reopen_after_metric_change_uses_identity_metric(tmp_path) -> None:
    x, y, model = _model(tmp_path)
    model.metric.set_weights(np.array([100.0, 1.0, 0.01]))
    model.persist_index_to_disk()
    del model
    reopened = EpistemicNearestNeighbors(
        np.zeros((0, 3)),
        np.zeros((0, 1)),
        metric_learning=ENNMetricLearning.AUTO,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=tmp_path,
        enn_storage=ENNStorage.DISK,
    )
    q = np.array([[0.5, 0.5, 0.5]])
    np.testing.assert_array_equal(reopened.neighbors(q, 5), _exact(x, q, np.ones(3), 5))


def test_auto_rule_needs_positive_finite_heldout_gain() -> None:
    assert AUTO_MIN_HELDOUT_GAIN == 0.0
    assert auto_uses_learned_metric(0.01) is True
    for gain in (0.0, -0.5, -np.inf, np.nan, np.inf):
        assert auto_uses_learned_metric(gain) is False


def _stream(tmp_path, f, n: int, batch: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 3))
    y = (f(x) + 0.05 * rng.standard_normal(n))[:, None]
    model = EpistemicNearestNeighbors(
        x[:batch],
        y[:batch],
        metric_learning=ENNMetricLearning.AUTO,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=tmp_path,
        enn_storage=ENNStorage.DISK,
    )
    counts = [model.metric.num_refits]
    for lo in range(batch, n, batch):
        model.add(x[lo : lo + batch], y[lo : lo + batch])
        counts.append(model.metric.num_refits)
    return x, model, counts


def test_auto_refits_on_growth_schedule_and_applies_sobol_weights(tmp_path) -> None:
    x, model, counts = _stream(tmp_path, lambda x: np.sin(6 * x[:, 0]), 600)
    metric = model.metric
    assert AUTO_REFIT_GROWTH == 1.5 and AUTO_RESERVOIR_CAPACITY == 1000
    assert counts == [0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5]
    assert metric.uses_learned_metric and metric.heldout_gain > 0
    assert metric.weights[0] > 1e3 * metric.weights[1:].max()
    q = np.array([[0.3, 0.6, 0.9]])
    np.testing.assert_array_equal(
        model.neighbors(q, 5), _exact(x, q, metric.weights, 5)
    )


def test_refit_growth_and_capacity_reach_the_rust_policy(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = rng.random((600, 3))
    y = np.sin(6 * x[:, 0])[:, None]
    model = EpistemicNearestNeighbors(
        x[:50],
        y[:50],
        metric_learning=ENNMetricLearning.AUTO,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=tmp_path,
        enn_storage=ENNStorage.DISK,
    )
    MBPANNMetric(model, refit_growth=3.0)
    counts = [model.metric.num_refits]
    for lo in range(50, 600, 50):
        model.add(x[lo : lo + 50], y[lo : lo + 50])
        counts.append(model.metric.num_refits)
    assert counts == [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]

    small = EpistemicNearestNeighbors(
        x[:5],
        y[:5],
        metric_learning=ENNMetricLearning.AUTO,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=tmp_path / "small",
        enn_storage=ENNStorage.DISK,
    )
    MBPANNMetric(small, reservoir_capacity=8, seed=3)
    with pytest.raises(ValueError, match="reservoir_capacity"):
        MBPANNMetric(small, reservoir_capacity=2)
    with pytest.raises(ValueError, match="tied_dims"):
        MBPANNMetric(small, tied_dims=((0, 1),))


def test_auto_keeps_identity_metric_like_none_when_gain_is_not_positive(
    tmp_path,
) -> None:
    x, model, _ = _stream(tmp_path, lambda z: np.zeros(len(z)), 300)
    metric = model.metric
    assert not metric.uses_learned_metric
    q = np.array([[0.3, 0.6, 0.9]])
    np.testing.assert_array_equal(model.neighbors(q, 5), _exact(x, q, np.ones(3), 5))
