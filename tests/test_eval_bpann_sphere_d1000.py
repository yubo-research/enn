from __future__ import annotations

from types import SimpleNamespace

import pytest

from enn.turbo.config.enn_index_driver import ENNIndexDriver
from evals import eval_bpann_sphere_d1000 as mod
from evals import flat_sphere as fs
from ops.stress import MeanSE


def test_evaluate_uses_bpann_d1000_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[SimpleNamespace] = []

    def fake_run(config: fs.FlatSphereConfig) -> fs.FlatSphereAggregate:
        calls.append(SimpleNamespace(config=config))
        return fs.FlatSphereAggregate(
            num_dim=1000,
            num_obs=10000,
            num_test=100,
            num_seeds=30,
            seed=0,
            loglik=MeanSE(mean=-1.2, se=0.1),
            nrmse=MeanSE(mean=0.5, se=0.02),
            rcorr=MeanSE(mean=0.9, se=0.01),
        )

    monkeypatch.setattr(fs, "run_flat_sphere_over_seeds", fake_run)
    mod.evaluate()
    out = capsys.readouterr().out
    assert len(calls) == 1
    cfg = calls[0].config
    assert cfg.num_dim == 1000
    assert cfg.num_obs == 10000
    assert cfg.num_test == 100
    assert cfg.num_seeds == 30
    assert cfg.index_driver == ENNIndexDriver.BPANN_DISK
    assert cfg.work_dir is not None
    assert "index_driver=BPANN_DISK" in out
    assert "EVAL: n = 10000 LARGER(loglik) = -1.2 ± 0.1 SMALLER(nrmse) = 0.5 ± 0.02 LARGER(rcorr) = 0.9 ± 0.01" in out
