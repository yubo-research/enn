from __future__ import annotations

import tempfile

import numpy as np
import pytest

from enn.enn.mbpann import MBPANNMetric
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_x_scaling import ENNXScaling
from evals import metric_12d as mod
from evals.short import eval_metric_12d as entry

TINY = mod.Metric12dConfig(
    n_grid=(10, 30),
    num_test=20,
    batch=8,
    k=4,
    num_fit_candidates=6,
    num_fit_samples=4,
    metric_fit_subsample=25,
)


def test_make_data_shapes_and_signal() -> None:
    x, y = mod.make_data(50, np.random.default_rng(0))
    assert x.shape == (50, mod.NUM_DIM)
    assert y.shape == (50, 1)
    assert np.all((x >= 0) & (x < 1))
    f = np.sin(6 * np.pi * x[:, 0]) + np.sin(6 * np.pi * x[:, 1])
    assert np.max(np.abs(y[:, 0] - f)) < 1.0


def test_build_model_drivers_and_x_scaling() -> None:
    x, y = mod.make_data(12, np.random.default_rng(1))
    with tempfile.TemporaryDirectory() as work_dir:
        flat = mod.build_model("flat", x, y, work_dir)
        scaled = mod.build_model("flat_scale_x", x, y, work_dir)
        bpann = mod.build_model("bpann_disk", x, y, work_dir)
        learned = mod.build_model("bpann_disk_metric_learning", x, y, work_dir)
        assert flat._index_driver == ENNIndexDriver.FLAT and flat.x_scaling == ENNXScaling.NONE
        assert scaled._index_driver == ENNIndexDriver.FLAT and scaled.x_scaling == ENNXScaling.SCALE_X
        assert bpann._index_driver == ENNIndexDriver.BPANN_DISK and bpann.x_scaling == ENNXScaling.NONE
        assert learned._index_driver == ENNIndexDriver.BPANN_DISK
        assert learned.x_scaling == ENNXScaling.METRIC_LEARNING
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
        theta = mod.fit_metric(helper, x, y, TINY, np.random.default_rng(3))
        assert helper.num_rescales + helper.num_rebuilds == 1
        assert not np.allclose(helper.weights, 1.0)
        np.testing.assert_allclose(np.exp(theta[:-1]), helper.weights)


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
        theta = mod.fit_metric(helper, x, y, TINY, np.random.default_rng(3), prev)
    assert calls[0][1] == {}
    np.testing.assert_array_equal(calls[1][0], prev)
    assert calls[1][1] == {"outer": 1, "restart": False}
    np.testing.assert_array_equal(theta, prev)


@pytest.mark.parametrize(("warm_start", "expected"), [(True, [False, False, True]), (False, [False] * 3)])
def test_advance_warm_starts_only_after_full_subsample(
    monkeypatch: pytest.MonkeyPatch, warm_start: bool, expected: list[bool]
) -> None:
    warm: list[bool] = []

    def fake_fit(helper, x, y, config, rng, prev_theta=None):
        warm.append(prev_theta is not None)
        return np.zeros(mod.NUM_DIM + 1)

    monkeypatch.setattr(mod, "fit_metric", fake_fit)
    monkeypatch.setattr(mod, "enn_fit", lambda *a, **kw: None)
    cfg = mod.Metric12dConfig(batch=8, metric_fit_subsample=25, metric_warm_start=warm_start)
    x, y = mod.make_data(60, np.random.default_rng(4))
    with tempfile.TemporaryDirectory() as work_dir:
        streamed = mod.StreamedModel("bpann_disk_metric_learning", work_dir, cfg)
        for lo, hi in ((0, 10), (10, 30), (30, 60)):
            streamed.advance(x, y, lo, hi)
    assert warm == expected


def test_format_eval_line() -> None:
    line = mod.format_eval_line(
        mod.CheckpointResult("flat", 30, -1.25, 0.5, 0.01, 0.002)
    )
    assert line == (
        "EVAL: model = flat n = 30 LARGER(loglik) = -1.2500 SMALLER(nrmse) = 0.5000 "
        "SMALLER(add_s) = 0.0100 SMALLER(query_s) = 0.0020"
    )


def test_run_eval_tiny_all_models(capsys: pytest.CaptureFixture[str]) -> None:
    results = mod.run_eval(TINY)
    out = capsys.readouterr().out
    assert [(r.model, r.num_obs) for r in results] == [
        (m, n) for m in mod.MODELS for n in TINY.n_grid
    ]
    for r in results:
        assert np.isfinite(r.loglik) and np.isfinite(r.nrmse)
        assert r.add_s > 0 and r.query_s > 0
    assert out.count("EVAL: model = ") == len(results)
    assert "EVAL: model = bpann_disk_metric_learning n = 30 " in out


def test_only_metric_learning_model_gets_metric_helper() -> None:
    assert mod.BPANN_X_SCALING == {
        "bpann_disk": ENNXScaling.NONE,
        "bpann_disk_metric_learning": ENNXScaling.METRIC_LEARNING,
    }
    x, y = mod.make_data(12, np.random.default_rng(5))
    with tempfile.TemporaryDirectory() as work_dir:
        helpers = {}
        for name in mod.BPANN_X_SCALING:
            streamed = mod.StreamedModel(name, work_dir, TINY)
            streamed._add_rows(x, y)
            helpers[name] = streamed.helper
    assert helpers["bpann_disk"] is None
    assert isinstance(helpers["bpann_disk_metric_learning"], MBPANNMetric)


def test_run_eval_rejects_bad_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        mod.run_eval(mod.Metric12dConfig(n_grid=(30, 10)))
    with pytest.raises(ValueError, match="strictly increasing"):
        mod.run_eval(mod.Metric12dConfig(n_grid=(1, 10)))


def test_evaluate_entry_invokes_run_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(entry, "run_eval", lambda: called.append(1))
    entry.evaluate()
    assert called == [1]
