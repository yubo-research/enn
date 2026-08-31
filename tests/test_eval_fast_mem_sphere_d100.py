from __future__ import annotations

from types import SimpleNamespace

import pytest

from enn.turbo.config.enn_index_driver import ENNIndexDriver
from evals import eval_fast_mem_sphere_d100 as mod
from evals import flat_sphere as fs
from ops.stress import MeanSE


def test_evaluate_uses_fast_mem_d100_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[SimpleNamespace] = []

    def fake_run(config: fs.FlatSphereConfig) -> fs.FlatSphereAggregate:
        calls.append(SimpleNamespace(config=config))
        return fs.FlatSphereAggregate(
            num_dim=100,
            num_obs=1000,
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
    assert cfg.num_dim == 100
    assert cfg.num_obs == 1000
    assert cfg.num_test == 100
    assert cfg.num_seeds == 30
    assert cfg.index_driver == ENNIndexDriver.FAST_MEM
    assert "num_dim=100 num_obs=1000 num_test=100" in out
    assert "index_driver=FAST_MEM" in out
    assert "EVAL: n = 1000 LARGER(loglik) = -1.2 ± 0.1 SMALLER(nrmse) = 0.5 ± 0.02 LARGER(rcorr) = 0.9 ± 0.01" in out
