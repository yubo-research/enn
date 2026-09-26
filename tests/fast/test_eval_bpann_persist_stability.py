from __future__ import annotations

import tempfile

import numpy as np
import pytest

from enn.turbo.config.enn_index_driver import ENNIndexDriver
from evals import bpann_persist_stability as mod
from evals.short import eval_bpann_persist_stability as entry


def test_make_synthetic_shapes() -> None:
    rng = np.random.default_rng(0)
    x, y = mod.make_synthetic(7, num_dim=10, num_metrics=3, rng=rng)
    assert x.shape == (7, 10)
    assert y.shape == (7, 3)


def test_format_eval_line() -> None:
    line = mod.format_eval_line(10, 0.0, 1.25e-8, 0.0, 2.5e-8)
    assert line.startswith("EVAL: n = 10 ")
    assert "SMALLER(abs_loocv_diff) = 0.0" in line
    assert "SMALLER(abs_rmse_diff) =" in line
    assert "SMALLER(abs_loocv_load_diff) = 0.0" in line
    assert "SMALLER(abs_rmse_load_diff) =" in line


def test_run_inner_persist_deltas_finite() -> None:
    with tempfile.TemporaryDirectory(prefix="enn_persist_inner_") as work_dir:
        d_loo, d_rmse, d_loo_load, d_rmse_load = mod.run_inner(
            mod.PersistConfig(
                num_obs=30,
                work_dir=work_dir,
                seed=0,
                k=5,
                num_fit_candidates=8,
                num_fit_samples=4,
                num_loo=100,
                num_test=20,
            )
        )
    assert d_loo >= 0.0
    assert d_rmse >= 0.0
    assert d_loo_load >= 0.0
    assert d_rmse_load >= 0.0
    assert np.isfinite(d_loo)
    assert np.isfinite(d_rmse)
    assert np.isfinite(d_loo_load)
    assert np.isfinite(d_rmse_load)


def test_run_inner_uses_bpann_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    sizes: list[int] = []

    real_build = mod.build_model

    def wrapped(x, y, work_dir):
        model = real_build(x, y, work_dir)
        seen.append(model._index_driver)
        real_add = model.add

        def add_and_track(x_add, y_add, yvar=None):
            real_add(x_add, y_add, yvar)
            sizes.append(len(model))

        model.add = add_and_track
        return model

    monkeypatch.setattr(mod, "build_model", wrapped)
    with tempfile.TemporaryDirectory(prefix="enn_persist_driver_") as work_dir:
        mod.run_inner(
            mod.PersistConfig(
                num_obs=12,
                work_dir=work_dir,
                seed=1,
                k=3,
                num_fit_candidates=5,
                num_fit_samples=3,
                num_loo=100,
                num_test=8,
            )
        )
    assert seen == [ENNIndexDriver.BPANN_DISK, ENNIndexDriver.BPANN_DISK]
    assert sizes == [12]


def test_run_inner_o0_is_training_prefix() -> None:
    with tempfile.TemporaryDirectory(prefix="enn_persist_o0_") as work_dir:
        d_loo, d_rmse, d_loo_load, d_rmse_load = mod.run_inner(
            mod.PersistConfig(
                num_obs=10,
                work_dir=work_dir,
                seed=2,
                k=3,
                num_fit_candidates=5,
                num_fit_samples=3,
                num_loo=100,
                num_test=8,
            )
        )
    assert np.isfinite(d_loo)
    assert np.isfinite(d_rmse)
    assert np.isfinite(d_loo_load)
    assert np.isfinite(d_rmse_load)


def test_run_inner_creates_side_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[str] = []
    real_build = mod.build_model

    def wrapped(x, y, work_dir):
        builds.append(work_dir)
        return real_build(x, y, work_dir)

    monkeypatch.setattr(mod, "build_model", wrapped)
    with tempfile.TemporaryDirectory(prefix="enn_persist_side_") as work_dir:
        _ = mod.run_inner(
            mod.PersistConfig(
                num_obs=12,
                work_dir=work_dir,
                seed=3,
                k=3,
                num_fit_candidates=5,
                num_fit_samples=3,
                num_loo=100,
                num_test=8,
            )
        )
    assert len(builds) == 2
    assert builds[0] == builds[1]


def test_run_eval_prints_grid(capsys: pytest.CaptureFixture[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="enn_persist_grid_") as work_dir:
        results = mod.run_eval(
            (10, 30),
            work_dir=work_dir,
            seed=0,
            k=4,
            num_fit_candidates=6,
            num_fit_samples=3,
        )
    out = capsys.readouterr().out
    assert len(results) == 2
    assert len(results[0]) == 5
    assert "EVAL: n = 10 " in out
    assert "EVAL: n = 30 " in out
    assert "SMALLER(abs_loocv_diff)" in out
    assert "SMALLER(abs_rmse_diff)" in out
    assert "SMALLER(abs_loocv_load_diff)" in out
    assert "SMALLER(abs_rmse_load_diff)" in out


def test_evaluate_entry_invokes_run_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []

    def fake_run_eval(**kwargs):
        called.append(1)
        return []

    monkeypatch.setattr(entry, "run_eval", fake_run_eval)
    entry.evaluate()
    assert called == [1]
