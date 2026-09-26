from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals import eval_flat_sphere_d10 as mod
from evals import flat_sphere as fs
from ops.stress import MeanSE


def test_gaussian_loglik_and_rmse() -> None:
    y = np.array([[0.0], [1.0]])
    mu = np.array([[0.0], [1.0]])
    se = np.array([[1.0], [1.0]])
    assert fs.gaussian_loglik(y, mu, se) == pytest.approx(-0.5 * np.log(2.0 * np.pi))
    assert fs.rmse(y, mu) == pytest.approx(0.0)
    assert fs.rmse(y, np.array([[1.0], [0.0]])) == pytest.approx(1.0)


def test_format_flat_sphere_eval_line() -> None:
    line = fs.format_flat_sphere_eval_line(
        fs.FlatSphereAggregate(
            num_dim=10,
            num_obs=10,
            num_test=100,
            num_seeds=30,
            seed=0,
            loglik=MeanSE(mean=-1.25, se=0.05),
            rmse=MeanSE(mean=0.8732, se=0.0123),
        )
    )
    assert line.startswith("EVAL: n = 10 ")
    assert "LARGER(loglik) = -1.25 ± 0.05" in line
    assert "SMALLER(rmse) = 0.8732 ± 0.0123" in line


def test_run_flat_sphere_seed_shapes() -> None:
    result = fs.run_flat_sphere_seed(
        fs.FlatSphereConfig(num_dim=3, num_obs=3, num_test=5, seed=0, k=2)
    )
    assert np.isfinite(result.loglik)
    assert np.isfinite(result.rmse)
    assert result.rmse >= 0.0


def test_run_flat_sphere_over_seeds_aggregates() -> None:
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
        )
    )
    assert agg.num_seeds == 2
    assert agg.num_obs == 3
    assert np.isfinite(agg.loglik.mean)
    assert np.isfinite(agg.loglik.se)
    assert np.isfinite(agg.rmse.mean)
    assert np.isfinite(agg.rmse.se)


def test_evaluate_uses_d10_defaults(
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
    assert "num_dim=10 num_obs=100 num_test=100" in out
    assert "index_driver=FLAT" in out
    assert "EVAL: n = 100 LARGER(loglik) = -1.2 ± 0.1 SMALLER(rmse) = 0.5 ± 0.02" in out
