from __future__ import annotations

import numpy as np
import pytest

from evals import metric_small_n as mod
from evals.metric_12d import Metric12dConfig
from evals.short import eval_metric_small_n as entry

TINY = mod.SmallNConfig(
    functions=("lin3", "sphere"),
    seeds=(0,),
    stream=Metric12dConfig(
        n_grid=(10, 120),
        num_test=20,
        batch=40,
        k=4,
        num_fit_candidates=6,
        num_fit_samples=4,
    ),
)


def test_make_data_functions() -> None:
    for name, f in mod.FUNCTIONS.items():
        x, y = mod.make_data(name, 50, np.random.default_rng(0))
        assert x.shape == (50, mod.NUM_DIM) and y.shape == (50, 1)
        assert np.max(np.abs(y[:, 0] - f(x))) < 1.0
    x = np.full((1, mod.NUM_DIM), 0.5)
    assert mod.FUNCTIONS["sphere"](x)[0] == 0.0
    assert mod.FUNCTIONS["lin3"](x)[0] == pytest.approx(1.0)


def test_format_eval_line() -> None:
    r = mod.SmallNResult("lin3", 2, "bpann_disk_auto", 30, -1.25, 0.5, 0.125, True)
    assert mod.format_eval_line(r) == (
        "EVAL: function = lin3 seed = 2 model = bpann_disk_auto n = 30 LARGER(loglik) = -1.2500 "
        "SMALLER(nrmse) = 0.5000 gain = 0.1250 on = 1"
    )


def test_run_eval_tiny(capsys: pytest.CaptureFixture[str]) -> None:
    results = mod.run_eval(TINY)
    out = capsys.readouterr().out
    assert [(r.function, r.model, r.num_obs) for r in results] == [
        (f, m, n) for f in TINY.functions for m in mod.MODELS for n in TINY.stream.n_grid
    ]
    for r in results:
        assert np.isfinite(r.loglik) and np.isfinite(r.nrmse)
        refit = r.model == "bpann_disk_auto" and r.num_obs >= 100
        assert (r.gain is not None) == refit and (r.on is not None) == refit
    assert out.count("EVAL: function = ") == len(results)
    assert out.count("SUMMARY: ") == len(TINY.functions) * len(TINY.stream.n_grid)
    assert "auto_on=" in out


def test_summarize_means_and_on_fraction() -> None:
    rows = [
        mod.SmallNResult("lin3", 0, "bpann_disk", 10, -1.0, 1.0, None, None),
        mod.SmallNResult("lin3", 1, "bpann_disk", 10, -2.0, 1.0, None, None),
        mod.SmallNResult("lin3", 0, "bpann_disk_auto", 10, 0.0, 1.0, 0.2, True),
        mod.SmallNResult("lin3", 1, "bpann_disk_auto", 10, -1.0, 1.0, -0.2, False),
    ]
    assert mod.summarize(rows) == [
        "SUMMARY: function=lin3 n=10 mean_loglik bpann_disk=-1.500 bpann_disk_auto=-0.500 auto_on=1/2"
    ]


def test_evaluate_entry_invokes_run_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(entry, "run_eval", lambda: called.append(1))
    entry.evaluate()
    assert called == [1]
