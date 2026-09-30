from __future__ import annotations

import numpy as np
import pytest

from evals import metric_12d
from evals import metric_ranges as mod
from evals.long import eval_metric_ranges as entry
from evals.short import eval_metric_ranges as short_entry

TINY = metric_12d.Metric12dConfig(
    n_grid=(10, 30),
    num_test=20,
    batch=8,
    k=4,
    num_fit_candidates=6,
    num_fit_samples=4,
    num_seeds=1,
)


def test_ranges_span_four_decades() -> None:
    assert mod.RANGES.shape == (mod.NUM_DIM,)
    assert mod.RANGES[0] == pytest.approx(1e-2)
    assert mod.RANGES[-1] == pytest.approx(1e2)
    assert np.all(np.diff(mod.RANGES) > 0)


def test_make_data_ranges_and_signal() -> None:
    x, y = mod.make_data(2000, np.random.default_rng(0))
    assert x.shape == (2000, mod.NUM_DIM) and y.shape == (2000, 1)
    u = x / mod.RANGES
    assert np.all((u >= 0) & (u < 1))
    assert np.allclose(x.max(axis=0) / mod.RANGES, 1.0, atol=0.01)
    f = (u - 0.5).sum(axis=1)
    assert np.max(np.abs(y[:, 0] - f)) < 1.0
    assert np.std(y) == pytest.approx(np.sqrt(1.0 + mod.NOISE_STD**2), rel=0.1)


def test_run_eval_uses_ranges_data(capsys: pytest.CaptureFixture[str]) -> None:
    summaries = mod.run_eval(TINY)
    out = capsys.readouterr().out
    assert out.startswith("ranges=0.01,")
    assert {s.model for s in summaries} == set(metric_12d.MODELS)
    assert {s.num_obs for s in summaries} == {10, 30}
    assert out.count("EVAL: ") == len(metric_12d.MODELS) * 2


def test_run_eval_passes_ranges_data_to_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(
        mod,
        "run_stream_eval",
        lambda config, models, data: seen.append((config, models, data)) or [],
    )
    assert mod.run_eval(TINY) == []
    assert mod.run_eval(TINY, models=("bpann_disk",)) == []
    assert seen == [(TINY, metric_12d.MODELS, mod.make_data), (TINY, ("bpann_disk",), mod.make_data)]


def test_evaluate_entry_invokes_run_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(entry, "run_eval", lambda: called.append(1))
    entry.evaluate()
    assert called == [1]


def test_short_entry_runs_long_config_up_to_1e5(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(short_entry, "run_eval", seen.append)
    short_entry.evaluate()
    assert seen == [
        metric_12d.Metric12dConfig(
            n_grid=(10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000), num_rows=1000000
        )
    ]
