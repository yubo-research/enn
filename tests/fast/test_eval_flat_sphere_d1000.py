from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals import eval_flat_sphere_d1000 as mod
from evals import flat_sphere as fs
from ops.stress import MeanSE


def test_evaluate_uses_d1000_defaults(
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
            rmse=MeanSE(mean=0.5, se=0.02),
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
    assert "num_dim=1000 num_obs=10000 num_test=100" in out
    assert "index_driver=FLAT" in out
    assert "EVAL: n = 10000 LARGER(loglik) = -1.2 ± 0.1 SMALLER(rmse) = 0.5 ± 0.02" in out
