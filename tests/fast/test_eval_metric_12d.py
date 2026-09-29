from __future__ import annotations

import tempfile
from dataclasses import replace

import numpy as np
import pytest

from enn.enn.mbpann import MBPANNMetric
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_x_scaling import ENNMetricLearning, ENNScaleX
from evals import metric_12d as mod
from evals.long import eval_metric_12d as entry
from evals.short import eval_metric_12d as short_entry
from ops.stress import MeanSE

TINY = mod.Metric12dConfig(
    n_grid=(10, 30),
    num_test=20,
    batch=8,
    k=4,
    num_fit_candidates=6,
    num_fit_samples=4,
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
        auto = mod.build_model("bpann_disk_auto", x, y, work_dir)
        bpann_scaled = mod.build_model("bpann_disk_scale_x", x, y, work_dir)
        assert bpann_scaled._index_driver == ENNIndexDriver.BPANN_DISK
        assert bpann_scaled.scale_x == ENNScaleX.ON
        assert bpann_scaled.metric_learning == ENNMetricLearning.NONE
        assert bpann.scale_x == ENNScaleX.OFF
        assert flat._index_driver == ENNIndexDriver.FLAT and flat.scale_x == ENNScaleX.OFF
        assert flat.metric_learning == ENNMetricLearning.NONE
        assert scaled._index_driver == ENNIndexDriver.FLAT and scaled.scale_x == ENNScaleX.ON
        assert bpann._index_driver == ENNIndexDriver.BPANN_DISK
        assert bpann.metric_learning == ENNMetricLearning.NONE
        assert auto._index_driver == ENNIndexDriver.BPANN_DISK
        assert auto.metric_learning == ENNMetricLearning.AUTO
        assert isinstance(auto.metric, MBPANNMetric) and bpann.metric is None
        with pytest.raises(ValueError, match="unknown model"):
            mod.build_model("nope", x, y, work_dir)


def test_advance_adds_rows_in_batches_and_auto_refits_inside_add(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "enn_fit", lambda *a, **kw: None)
    cfg = mod.Metric12dConfig(batch=40)
    x, y = mod.make_data(260, np.random.default_rng(4))
    with tempfile.TemporaryDirectory() as work_dir:
        streamed = mod.StreamedModel("bpann_disk_auto", work_dir, cfg)
        for lo, hi in ((0, 30), (30, 100), (100, 260)):
            assert streamed.advance(x, y, lo, hi) > 0
        assert len(streamed.model) == 260
        assert streamed.model.metric.reservoir.num_seen == 260
        assert streamed.model.metric.num_refits == 2


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


def test_bpann_models_and_metric_learning_modes() -> None:
    assert mod.MODELS == ("flat", "flat_scale_x", "bpann_disk", "bpann_disk_auto")
    assert mod.BPANN_METRIC_LEARNING == {
        "bpann_disk": ENNMetricLearning.NONE,
        "bpann_disk_scale_x": ENNMetricLearning.NONE,
        "bpann_disk_auto": ENNMetricLearning.AUTO,
    }


def test_run_eval_rejects_bad_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        mod.run_eval(mod.Metric12dConfig(n_grid=(30, 10)))
    with pytest.raises(ValueError, match="strictly increasing"):
        mod.run_eval(mod.Metric12dConfig(n_grid=(1, 10)))
    with pytest.raises(ValueError, match="num_seeds"):
        mod.run_eval(mod.Metric12dConfig(num_seeds=0))
    with pytest.raises(ValueError, match="num_rows"):
        mod.run_eval(mod.Metric12dConfig(n_grid=(10, 30), num_rows=20))


def test_evaluate_entry_invokes_run_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(entry, "run_eval", lambda: called.append(1))
    entry.evaluate()
    assert called == [1]


def test_short_entry_runs_long_config_up_to_1e5(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[mod.Metric12dConfig] = []
    monkeypatch.setattr(short_entry, "run_eval", seen.append)
    short_entry.evaluate()
    assert seen == [
        mod.Metric12dConfig(
            n_grid=(10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000), num_rows=1000000
        )
    ]


def test_truncated_grid_with_same_num_rows_reproduces_long_prefix() -> None:
    short = mod.run_eval(replace(TINY, num_rows=100), models=("flat",))
    long = mod.run_eval(replace(TINY, n_grid=(10, 30, 100)), models=("flat",))
    assert [(r.num_obs, r.loglik, r.nrmse) for r in short] == [
        (r.num_obs, r.loglik, r.nrmse) for r in long[:2]
    ]
