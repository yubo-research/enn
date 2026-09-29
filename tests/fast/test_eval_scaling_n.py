from __future__ import annotations

import math

import numpy as np
import pytest

from evals import scaling_n as mod
from evals.scaling_fit import TERMS, RegFit, TermTest, best_term
from evals.short import eval_scaling
from ops.stress import MeanSE

TINY = mod.ScalingConfig(
    n_grid=(20, 40, 60, 80),
    num_test=10,
    batch=20,
    k=4,
    num_fit_candidates=4,
    num_fit_samples=4,
    num_seeds=2,
    isolate=False,
)


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
        {"n": 10.0, **{k: 1.0 for k in mod.METRICS}},
        {"n": 10.0, **{k: 3.0 for k in mod.METRICS}},
        {"n": 100.0, **{k: 5.0 for k in mod.METRICS}},
    ]
    summary = mod.summarize(rows)
    assert list(summary) == [10, 100]
    assert summary[10]["add_s"] == MeanSE(mean=2.0, se=1.0)
    line = mod.format_eval_line(10, summary[10])
    assert line.startswith("EVAL: model = bpann_disk_auto_ols n = 10 SMALLER(rss_mib) = 2.0000 ± 1.0000 ")
    assert "LARGER(loglik) = 2.0000 ± 1.0000" in line and "SMALLER(nrmse)" in line
    assert all(f"({k})" in line for k in mod.METRICS)
    assert mod.format_seed_line(3, {"n": 100.0, "add_s": 5.0}) == "seed = 3 n = 100 add_s = 5.0000"


def test_format_reg_line() -> None:
    fits = {
        "lnN": RegFit(1.0, {"lnN": TermTest(0.5, 2.0, 0.06)}, 0.5, 21),
        "N": RegFit(1.5, {"N": TermTest(0.25, 4.0, 0.001)}, 0.9, 21),
    }
    assert mod.format_reg_line("rss_mib", fits) == (
        "EVAL: model = bpann_disk_auto_ols reg = rss_mib obs = 21 best = N "
        "r2_lnN = 0.5000 b0_lnN = 1 b_lnN = 0.5 t_lnN = 2 p_lnN = 0.06 "
        "r2_N = 0.9000 b0_N = 1.5 b_N = 0.25 t_N = 4 p_N = 0.001"
    )
    assert "obs = 0 best = none" in mod.format_reg_line("q", {})


def test_regress_recovers_linear_memory() -> None:
    rows = [{"n": float(n), **{m: 3.0 + 0.002 * n + 0.01 * s for m in mod.REG_METRICS}} for n in (100, 1000, 10000, 100000) for s in range(3)]
    fits = mod.regress(rows)
    assert set(fits) == set(mod.REG_METRICS)
    term_fits = fits["rss_mib"]
    assert list(term_fits) == list(TERMS) and term_fits["N"].obs == 12
    assert best_term(term_fits) == "N"
    assert term_fits["N"].tests["N"].coef == pytest.approx(2.0, rel=1e-3)


def test_run_eval_tiny(capsys: pytest.CaptureFixture[str]) -> None:
    summary, fits = mod.run_eval(TINY)
    out = capsys.readouterr().out
    assert list(summary) == list(TINY.n_grid)
    for stats in summary.values():
        assert stats["disk_mib"].mean > 0 and stats["add_s"].mean > 0
        assert stats["query_s"].mean > 0 and np.isfinite(stats["nrmse"].mean)
    assert out.count("seed = ") == TINY.num_seeds * len(TINY.n_grid)
    assert out.count("EVAL: ") == len(TINY.n_grid) + len(mod.REG_METRICS)
    assert all(f"reg = {m} obs = 8 best = " in out for m in mod.REG_METRICS)
    assert set(fits) == set(mod.REG_METRICS)


def test_run_eval_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="increasing"):
        mod.run_eval(mod.ScalingConfig(n_grid=(100, 10)))
    with pytest.raises(ValueError, match="checkpoints"):
        mod.run_eval(mod.ScalingConfig(n_grid=(10, 100)))


def test_run_seed_isolated_matches_grid() -> None:
    cfg = mod.ScalingConfig(
        n_grid=(20, 40), num_test=5, batch=20, k=4, num_fit_candidates=2, num_fit_samples=2
    )
    rows = mod.run_seed_isolated(cfg)
    assert [r["n"] for r in rows] == [20.0, 40.0]
    assert all(r["disk_mib"] > 0 for r in rows)


def test_entry_invokes_run_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(eval_scaling, "run_eval", lambda: called.append(1))
    eval_scaling.evaluate()
    assert called == [1]
