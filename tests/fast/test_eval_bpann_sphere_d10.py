from __future__ import annotations

import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from enn.turbo.config.enn_index_driver import ENNIndexDriver
from evals import eval_bpann_sphere_d10 as mod
from evals import flat_sphere as fs
from ops.stress import MeanSE


def test_run_bpann_sphere_seed_shapes() -> None:
    with tempfile.TemporaryDirectory(prefix="enn_bpann_sphere_seed_") as work_dir:
        result = fs.run_flat_sphere_seed(
            fs.FlatSphereConfig(
                num_dim=3,
                num_obs=3,
                num_test=5,
                seed=0,
                k=2,
                index_driver=ENNIndexDriver.BPANN_DISK,
                work_dir=work_dir,
            )
        )
    assert np.isfinite(result.loglik)
    assert np.isfinite(result.rmse)
    assert result.rmse >= 0.0


def test_run_bpann_sphere_over_seeds_aggregates() -> None:
    with tempfile.TemporaryDirectory(prefix="enn_bpann_sphere_agg_") as work_dir:
        agg = fs.run_flat_sphere_over_seeds(
            fs.FlatSphereConfig(
                num_dim=3,
                num_obs=3,
                num_test=5,
                seed=0,
                num_seeds=2,
                k=2,
                num_fit_candidates=5,
                num_fit_samples=3,
                index_driver=ENNIndexDriver.BPANN_DISK,
                work_dir=work_dir,
            )
        )
    assert agg.num_seeds == 2
    assert agg.num_obs == 3
    assert np.isfinite(agg.loglik.mean)
    assert np.isfinite(agg.loglik.se)
    assert np.isfinite(agg.rmse.mean)
    assert np.isfinite(agg.rmse.se)


def test_bpann_sphere_requires_work_dir() -> None:
    with pytest.raises(ValueError, match="bpann_disk requires work_dir"):
        fs.run_flat_sphere_seed(
            fs.FlatSphereConfig(
                num_dim=3,
                num_obs=3,
                num_test=5,
                seed=0,
                k=2,
                index_driver=ENNIndexDriver.BPANN_DISK,
                work_dir=None,
            )
        )


def test_evaluate_uses_bpann_d10_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[SimpleNamespace] = []

    def fake_run(config: fs.FlatSphereConfig) -> fs.FlatSphereAggregate:
        calls.append(SimpleNamespace(config=config))
        return fs.FlatSphereAggregate(
            num_dim=10,
            num_obs=100,
            num_test=100,
            num_seeds=30,
            seed=0,
            loglik=MeanSE(mean=-1.2, se=0.1),
            rmse=MeanSE(mean=0.5, se=0.02),
        )

    monkeypatch.setattr(fs, "run_flat_sphere_over_seeds", fake_run)
    mod.evaluate()
    out = capsys.readouterr().out
    assert len(calls) == 1
    cfg = calls[0].config
    assert cfg.num_dim == 10
    assert cfg.num_obs == 100
    assert cfg.num_test == 100
    assert cfg.num_seeds == 30
    assert cfg.index_driver == ENNIndexDriver.BPANN_DISK
    assert cfg.work_dir is not None
    assert "index_driver=BPANN_DISK" in out
    assert "EVAL: n = 100 LARGER(loglik) = -1.2 ± 0.1 SMALLER(rmse) = 0.5 ± 0.02" in out
