from __future__ import annotations

import math

import numpy as np
import pytest

from evals import scaling_n as mod
from evals.short import eval_scaling_add, eval_scaling_memory, eval_scaling_query
from ops.stress import MeanSE

TINY = mod.ScalingConfig(
    n_grid=(20, 60, 120),
    num_test=10,
    batch=20,
    k=4,
    num_fit_candidates=4,
    num_fit_samples=4,
    num_seeds=2,
    slope_min_n=60,
    isolate=False,
)


def test_loglog_slope() -> None:
    ns = [10, 100, 1000]
    assert mod.loglog_slope(ns, [1.0, 10.0, 100.0]) == pytest.approx(1.0)
    assert mod.loglog_slope(ns, [5.0, 5.0, 5.0]) == pytest.approx(0.0)
    assert math.isnan(mod.loglog_slope(ns, [-1.0, 0.0, 3.0]))
    assert math.isnan(mod.loglog_slope(ns, [math.nan, 1.0, 0.0]))


def test_memory_mib_and_reset_peak() -> None:
    big = np.ones(20_000_000)
    del big
    mod.reset_peak()
    mem = mod.memory_mib()
    assert set(mem) == {"rss_mib", "anon_mib", "file_mib", "peak_mib"}
    assert mem["rss_mib"] > 0 and mem["peak_mib"] < mem["rss_mib"] + 100
    assert mem["anon_mib"] + mem["file_mib"] <= mem["rss_mib"] + 1


def test_proc_status_missing_key_is_nan() -> None:
    assert math.isnan(mod._proc_status_kib("NoSuchKey"))


def test_dir_mib(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.bin").write_bytes(b"\0" * (1024 * 1024))
    assert mod.dir_mib(str(tmp_path)) == pytest.approx(1.0)


def test_summarize_and_format_lines() -> None:
    rows = [
        {"n": 10.0, "add_s": 1.0, "add_us": 2.0, "fit_s": 3.0},
        {"n": 10.0, "add_s": 3.0, "add_us": 2.0, "fit_s": 3.0},
        {"n": 100.0, "add_s": 5.0, "add_us": 2.0, "fit_s": 3.0},
    ]
    summary = mod.summarize(rows)
    assert list(summary) == [10, 100]
    assert summary[10]["add_s"] == MeanSE(mean=2.0, se=1.0)
    assert mod.format_eval_line("add", 10, summary[10]) == (
        "EVAL: model = bpann_disk_auto_ols n = 10 SMALLER(add_s) = 2.0000 ± 1.0000 "
        "SMALLER(add_us) = 2.0000 ± 0.0000 SMALLER(fit_s) = 3.0000 ± 0.0000"
    )
    assert mod.format_slope_line("add", summary, 10) == (
        "EVAL: model = bpann_disk_auto_ols n = 10..100 SMALLER(slope_add_us) = 0.000"
    )
    assert mod.format_seed_line(3, rows[2]) == (
        "seed = 3 n = 100 add_s = 5.0000 add_us = 2.0000 fit_s = 3.0000"
    )


@pytest.mark.parametrize("kind", sorted(mod.METRICS))
def test_run_eval_tiny(kind: str, capsys: pytest.CaptureFixture[str]) -> None:
    summary = mod.run_eval(kind, TINY)
    out = capsys.readouterr().out
    assert list(summary) == list(TINY.n_grid)
    for stats in summary.values():
        assert stats["disk_mib"].mean > 0 and stats["add_s"].mean > 0
        assert stats["query_s"].mean > 0 and np.isfinite(stats["nrmse"].mean)
    assert out.count("seed = ") == TINY.num_seeds * len(TINY.n_grid)
    assert out.count("EVAL: ") == len(TINY.n_grid) + 1
    assert f"SMALLER({mod.METRICS[kind][0]})" in out
    assert f"n = 60..120 SMALLER(slope_{mod.SLOPE_METRICS[kind][0]})" in out


def test_run_eval_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="kind"):
        mod.run_eval("speed", TINY)
    with pytest.raises(ValueError, match="increasing"):
        mod.run_eval("add", mod.ScalingConfig(n_grid=(100, 10)))
    with pytest.raises(ValueError, match="slope_min_n"):
        mod.run_eval("add", mod.ScalingConfig(n_grid=(10, 100), slope_min_n=50))


def test_run_seed_isolated_matches_grid() -> None:
    cfg = mod.ScalingConfig(
        n_grid=(20, 40), num_test=5, batch=20, k=4, num_fit_candidates=2, num_fit_samples=2
    )
    rows = mod.run_seed_isolated(cfg)
    assert [r["n"] for r in rows] == [20.0, 40.0]
    assert all(r["disk_mib"] > 0 for r in rows)


@pytest.mark.parametrize(
    ("entry", "kind"),
    [(eval_scaling_memory, "memory"), (eval_scaling_add, "add"), (eval_scaling_query, "query")],
)
def test_entries_invoke_run_eval(entry, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(entry, "run_eval", called.append)
    entry.evaluate()
    assert called == [kind]
