from __future__ import annotations

import tempfile

import numpy as np
import pytest

from enn.enn.mbpann import MBPANNMetric
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_x_scaling import ENNMetricLearning, ENNScaleX
from evals import metric_12d as mod
from evals.metric_stream_models import STREAM_METRIC_MODELS
from evals.short import eval_metric_12d as entry
from ops.stress import MeanSE

TINY = mod.Metric12dConfig(
    n_grid=(10, 30),
    num_test=20,
    batch=8,
    k=4,
    num_fit_candidates=6,
    num_fit_samples=4,
    metric_fit_subsample=25,
    num_seeds=2,
)


def test_default_config_uses_ten_seeds() -> None:
    assert mod.NUM_SEEDS == 10
    assert mod.Metric12dConfig().num_seeds == 10


def test_make_data_shapes_and_signal() -> None:
    x, y = mod.make_data(50, np.random.default_rng(0))
    assert x.shape == (50, mod.NUM_DIM)
    assert y.shape == (50, 1)
    assert np.all((x >= 0) & (x < 1))
    f = np.sin(6 * np.pi * x[:, 0]) + np.sin(6 * np.pi * x[:, 1])
    assert np.max(np.abs(y[:, 0] - f)) < 1.0


def test_build_model_drivers_scale_x_and_metric_learning() -> None:
    x, y = mod.make_data(12, np.random.default_rng(1))
    with tempfile.TemporaryDirectory() as work_dir:
        flat = mod.build_model("flat", x, y, work_dir)
        scaled = mod.build_model("flat_scale_x", x, y, work_dir)
        bpann = mod.build_model("bpann_disk", x, y, work_dir)
        learned = mod.build_model("bpann_disk_metric_learning", x, y, work_dir)
        bpann_scaled = mod.build_model("bpann_disk_scale_x", x, y, work_dir)
        assert bpann_scaled._index_driver == ENNIndexDriver.BPANN_DISK
        assert bpann_scaled.scale_x == ENNScaleX.ON
        assert bpann_scaled.metric_learning == ENNMetricLearning.OFF
        assert bpann.scale_x == ENNScaleX.OFF
        assert flat._index_driver == ENNIndexDriver.FLAT and flat.scale_x == ENNScaleX.OFF
        assert flat.metric_learning == ENNMetricLearning.OFF
        assert scaled._index_driver == ENNIndexDriver.FLAT and scaled.scale_x == ENNScaleX.ON
        assert bpann._index_driver == ENNIndexDriver.BPANN_DISK
        assert bpann.metric_learning == ENNMetricLearning.OFF
        assert learned._index_driver == ENNIndexDriver.BPANN_DISK
        assert learned.metric_learning == ENNMetricLearning.ON
        with pytest.raises(ValueError, match="unknown model"):
            mod.build_model("nope", x, y, work_dir)


def test_load_iaml_core_exposes_fitter() -> None:
    core = mod.load_iaml_core()
    assert callable(core.fit_exact)
    assert core.Metric(mod.NUM_DIM).a.shape == (mod.NUM_DIM,)


def test_fit_metric_sets_weights() -> None:
    x, y = mod.make_data(40, np.random.default_rng(2))
    with tempfile.TemporaryDirectory() as work_dir:
        model = mod.build_model("bpann_disk_metric_learning", x, y, work_dir)
        model.ensure_index_sync()
        helper = MBPANNMetric(model)
        fit = mod.fit_metric(helper, x, y, TINY, np.random.default_rng(3))
        assert helper.num_rescales + helper.num_rebuilds == 1
        assert not np.allclose(helper.weights, 1.0)
        assert fit.heldout_gain is None
        np.testing.assert_allclose(np.exp(fit.theta[:-1]), helper.weights)


def test_fit_metric_warm_start_uses_one_round_from_prev(monkeypatch: pytest.MonkeyPatch) -> None:
    core = mod.load_iaml_core()
    calls: list[tuple[np.ndarray, dict]] = []
    monkeypatch.setattr(
        core, "fit_exact", lambda metric, x, y, k, **kw: calls.append((metric.theta.copy(), kw))
    )
    x, y = mod.make_data(40, np.random.default_rng(2))
    prev = np.linspace(-1.0, 1.0, mod.NUM_DIM + 1)
    with tempfile.TemporaryDirectory() as work_dir:
        model = mod.build_model("bpann_disk_metric_learning", x, y, work_dir)
        model.ensure_index_sync()
        helper = MBPANNMetric(model)
        mod.fit_metric(helper, x, y, TINY, np.random.default_rng(3))
        fit = mod.fit_metric(helper, x, y, TINY, np.random.default_rng(3), prev)
    assert calls[0][1] == {}
    np.testing.assert_array_equal(calls[1][0], prev)
    assert calls[1][1] == {"outer": 1, "restart": False}
    np.testing.assert_array_equal(fit.theta, prev)


@pytest.mark.parametrize(("warm_start", "expected"), [(True, [False, False, True]), (False, [False] * 3)])
def test_advance_warm_starts_only_after_full_subsample(
    monkeypatch: pytest.MonkeyPatch, warm_start: bool, expected: list[bool]
) -> None:
    warm: list[bool] = []

    def fake_fit(helper, x, y, config, rng, prev_theta=None):
        warm.append(prev_theta is not None)
        return mod.MetricFit(theta=np.zeros(mod.NUM_DIM + 1))

    monkeypatch.setattr(mod, "fit_metric", fake_fit)
    monkeypatch.setattr(mod, "enn_fit", lambda *a, **kw: None)
    cfg = mod.Metric12dConfig(batch=8, metric_fit_subsample=25, metric_warm_start=warm_start)
    x, y = mod.make_data(60, np.random.default_rng(4))
    with tempfile.TemporaryDirectory() as work_dir:
        streamed = mod.StreamedModel("bpann_disk_metric_learning", work_dir, cfg)
        for lo, hi in ((0, 10), (10, 30), (30, 60)):
            streamed.advance(x, y, lo, hi)
    assert warm == expected


def test_format_eval_line_mean_pm_se() -> None:
    line = mod.format_eval_line(
        mod.CheckpointSummary(
            "flat",
            30,
            MeanSE(-1.25, 0.125),
            MeanSE(0.5, 0.02),
            MeanSE(0.01, 0.001),
            MeanSE(0.002, float("nan")),
        )
    )
    assert line == (
        "EVAL: model = flat n = 30 LARGER(loglik) = -1.2500 ± 0.1250 "
        "SMALLER(nrmse) = 0.5000 ± 0.0200 SMALLER(add_s) = 0.0100 ± 0.0010 "
        "SMALLER(query_s) = 0.0020"
    )


def test_format_seed_line_is_not_an_eval_line() -> None:
    line = mod.format_seed_line(3, mod.CheckpointResult("flat", 30, -1.25, 0.5, 0.01, 0.002))
    assert line == (
        "seed = 3 model = flat n = 30 loglik = -1.2500 nrmse = 0.5000 "
        "add_s = 0.0100 query_s = 0.0020"
    )


def test_summarize_mean_and_se_per_model_and_n() -> None:
    rows = [
        mod.CheckpointResult("a", 10, 1.0, 0.1, 1.0, 2.0),
        mod.CheckpointResult("a", 20, 5.0, 0.5, 1.0, 2.0),
        mod.CheckpointResult("a", 10, 3.0, 0.3, 3.0, 4.0),
        mod.CheckpointResult("a", 20, 5.0, 0.5, 1.0, 2.0),
    ]
    first, second = mod.summarize(rows)
    assert (first.model, first.num_obs, second.num_obs) == ("a", 10, 20)
    assert first.loglik == MeanSE(2.0, 1.0)
    assert first.add_s == MeanSE(2.0, 1.0)
    assert first.nrmse.mean == pytest.approx(0.2) and first.nrmse.se == pytest.approx(0.1)
    assert second.loglik == MeanSE(5.0, 0.0)


def test_run_eval_tiny_all_models(capsys: pytest.CaptureFixture[str]) -> None:
    results = mod.run_eval(TINY)
    out = capsys.readouterr().out
    assert [(r.model, r.num_obs) for r in results] == [
        (m, n) for m in mod.MODELS for n in TINY.n_grid
    ]
    for r in results:
        assert np.isfinite(r.loglik.mean) and np.isfinite(r.nrmse.mean)
        assert np.isfinite(r.loglik.se) and np.isfinite(r.nrmse.se)
        assert r.add_s.mean > 0 and r.query_s.mean > 0
    assert "num_seeds=2" in out
    assert out.count("EVAL: model = ") == len(results)
    assert out.count("\nseed = 0 model = ") == len(results)
    assert out.count("\nseed = 1 model = ") == len(results)
    assert out.count(" ± ") == 4 * len(results)
    assert "EVAL: model = bpann_disk_metric_learning n = 30 " in out
    assert "EVAL: model = bpann_disk_auto n = 30 " in out


def test_run_eval_streams_each_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, int, str]] = []

    def fake_run_model(name, work_dir, config, data):
        assert data is mod.make_data
        seen.append((name, config.seed, work_dir))
        return [mod.CheckpointResult(name, n, float(config.seed), 0.0, 1.0, 1.0) for n in config.n_grid]

    monkeypatch.setattr(mod, "run_model", fake_run_model)
    cfg = mod.Metric12dConfig(n_grid=(10, 30), seed=5, num_seeds=3)
    results = mod.run_eval(cfg, models=("flat", "bpann_disk"))
    assert [(name, seed) for name, seed, _ in seen] == [
        (m, s) for s in (5, 6, 7) for m in ("flat", "bpann_disk")
    ]
    assert len({work_dir for _, _, work_dir in seen}) == 3
    assert [(r.model, r.num_obs) for r in results] == [
        ("flat", 10), ("flat", 30), ("bpann_disk", 10), ("bpann_disk", 30)
    ]
    assert results[0].loglik == MeanSE(6.0, 1.0 / np.sqrt(3.0))


def test_only_learned_metric_models_get_metric_helper() -> None:
    assert mod.BPANN_METRIC_LEARNING == {
        "bpann_disk": ENNMetricLearning.OFF,
        "bpann_disk_metric_learning": ENNMetricLearning.ON,
        "bpann_disk_auto": ENNMetricLearning.AUTO,
        **{name: ENNMetricLearning.ON for name in STREAM_METRIC_MODELS},
    }
    assert mod.LEARNED_METRIC_MODELS == {
        "bpann_disk_metric_learning", "bpann_disk_auto", *STREAM_METRIC_MODELS
    }
    x, y = mod.make_data(12, np.random.default_rng(5))
    with tempfile.TemporaryDirectory() as work_dir:
        helpers = {}
        for name in mod.BPANN_METRIC_LEARNING:
            streamed = mod.StreamedModel(name, work_dir, TINY)
            streamed._add_rows(x, y)
            helpers[name] = streamed.helper
    assert helpers["bpann_disk"] is None
    assert isinstance(helpers["bpann_disk_metric_learning"], MBPANNMetric)
    assert isinstance(helpers["bpann_disk_auto"], MBPANNMetric)


def test_heldout_metric_gain_sign_tracks_signal() -> None:
    rng = np.random.default_rng(6)
    x = rng.random((80, mod.NUM_DIM))
    y_signal = x[:, 0] + 2.0 * x[:, 1] - x[:, 2] + 0.1 * rng.standard_normal(80)
    assert mod.heldout_metric_gain(x, y_signal, 10, np.random.default_rng(7)) > 0.5
    x_small, y_small = mod.make_data(10, np.random.default_rng(8))
    assert mod.heldout_metric_gain(x_small, y_small[:, 0], 10, np.random.default_rng(9)) < 0
    assert mod.heldout_metric_gain(x[:3], y_signal[:3], 10, rng) == float("-inf")


@pytest.mark.parametrize("gain", [-0.3, 0.4])
def test_fit_metric_auto_applies_metric_only_if_heldout_gain_positive(
    monkeypatch: pytest.MonkeyPatch, gain: float
) -> None:
    monkeypatch.setattr(mod, "heldout_metric_gain", lambda *a: gain)
    x, y = mod.make_data(40, np.random.default_rng(2))
    with tempfile.TemporaryDirectory() as work_dir:
        model = mod.build_model("bpann_disk_auto", x, y, work_dir)
        model.ensure_index_sync()
        helper = MBPANNMetric(model)
        fit = mod.fit_metric(helper, x, y, TINY, np.random.default_rng(3))
    assert fit.heldout_gain == gain
    if gain > 0:
        np.testing.assert_allclose(np.exp(fit.theta[:-1]), helper.weights)
        assert not np.allclose(helper.weights, 1.0)
    else:
        assert fit.theta is None
        np.testing.assert_array_equal(helper.weights, np.ones(mod.NUM_DIM))
        assert helper.num_rescales + helper.num_rebuilds == 0


def test_run_eval_rejects_bad_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        mod.run_eval(mod.Metric12dConfig(n_grid=(30, 10)))
    with pytest.raises(ValueError, match="strictly increasing"):
        mod.run_eval(mod.Metric12dConfig(n_grid=(1, 10)))
    with pytest.raises(ValueError, match="num_seeds"):
        mod.run_eval(mod.Metric12dConfig(num_seeds=0))


def test_evaluate_entry_invokes_run_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(entry, "run_eval", lambda: called.append(1))
    entry.evaluate()
    assert called == [1]
