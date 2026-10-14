from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np

from ops.qa_ts import (
    LOOKBACK,
    flatten_windows,
    nonlinear_signal,
    observation_grid,
    occupy,
    run_one,
    simulate_arma,
    simulate_nonlinear,
    simulate_periodic,
    timestep_counts,
)


def test_grid_matches_requested_sizes() -> None:
    assert observation_grid() == [100, 300, 1000, 3000, 10000, 30000, 100000]


def test_counts_sum_to_n_and_include_a_zero() -> None:
    counts = timestep_counts(100, 10, np.random.default_rng(0))
    assert int(counts.sum()) == 100
    assert counts.shape == (10,)
    assert int(counts[8]) == 0
    assert float(np.mean(counts)) == 10.0


def test_runs_reproduce_counts() -> None:
    counts = np.array([2, 2, 0, 1, 1], dtype=np.int64)
    cover = np.zeros(len(counts), dtype=int)
    for _sid, t0, length in occupy(counts):
        cover[t0 : t0 + length] += 1
    assert np.array_equal(cover, counts)


def test_flatten_windows_are_time_major() -> None:
    x = np.arange(8 * 10, dtype=float).reshape(8, 10)
    flats = flatten_windows(x)
    assert flats.shape[0] == 8 - LOOKBACK
    assert np.allclose(flats[0], x[:LOOKBACK].reshape(-1))


def test_arma_and_periodic_match_their_formulas() -> None:
    rng = np.random.default_rng(2)
    t0, length = 3, 6
    x, y = simulate_arma(t0, length, rng, 0.0)
    times = t0 + np.arange(length)
    expect = x.mean(axis=1) + 0.25 * np.sin(2.0 * np.pi * times / 30.0)
    assert np.allclose(y, expect)
    x, y = simulate_periodic(t0, length, np.random.default_rng(2), 0.0)
    phase = 2.0 * np.pi * times / 24.0
    expect = np.sin(phase) + 0.5 * np.cos(2.0 * phase) + 0.2 * x[:, 0]
    assert np.allclose(y, expect)


def test_nonlinear_target_uses_five_lags() -> None:
    rng = np.random.default_rng(1)
    x, y = simulate_nonlinear(0, 12, rng, 0.0)
    expect = np.tanh(x[4, 0] * x[3, 1] + x[2, 2]) + np.sin(x[1, 3] + x[0, 4])
    assert abs(float(y[5]) - float(expect)) < 1e-8
    assert nonlinear_signal(x[:6], np.zeros((8, x.shape[1]))).shape == (6,)


def test_run_one_scores_four_models() -> None:
    row = run_one("periodic", 100, 0, 0.1)
    assert row["n_test"] > 0
    assert row["n_train"] > 0
    for name in ("ennts", "ennts_metlrn", "enn", "bpann_auto"):
        assert np.isfinite(row[f"loglik_{name}"])
        assert np.isfinite(row[f"nrmse_{name}"])
        assert row[f"add_s_{name}"] >= 0.0
        assert row[f"query_s_{name}"] >= 0.0
        assert row[f"seconds_{name}"] + 1e-9 >= row[f"add_s_{name}"] + row[f"query_s_{name}"]


def test_compare_command_delegates_to_sweep(monkeypatch, tmp_path) -> None:
    from click.testing import CliRunner

    from ops.qa import cli

    seen: dict = {}

    def fake_sweep(kind, sizes, seed, out_dir):
        seen["kind"] = kind
        seen["sizes"] = list(sizes)
        seen["seed"] = seed
        seen["out"] = out_dir
        return []

    monkeypatch.setattr("ops.qa.run_sweep", fake_sweep)
    result = CliRunner().invoke(
        cli,
        ["compare", "--function", "arma", "--seed", "3", "--out", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert seen["kind"] == "arma"
    assert seen["sizes"] == [100, 300, 1000, 3000, 10000, 30000, 100000]
    assert seen["seed"] == 3
    assert seen["out"] == str(tmp_path)


_REPO = Path(__file__).resolve().parents[2]
_QA_TS = _REPO / "ops" / "qa_ts.py"


def _run_qa_ts(args: list[str], pythonpath: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, str(_QA_TS), *args],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_script_is_executable() -> None:
    mode = _QA_TS.stat().st_mode
    assert mode & stat.S_IXUSR
    assert _QA_TS.read_text(encoding="utf-8").startswith("#!/usr/bin/env python\n")


def test_help_prints_usage_without_numpy(tmp_path: Path) -> None:
    blocker = tmp_path / "numpy.py"
    blocker.write_text("raise RuntimeError('numpy imported during help')\n", encoding="utf-8")
    blocked = f"{tmp_path}{os.pathsep}{_REPO / 'src'}"
    result = _run_qa_ts(["--help"], blocked)
    assert result.returncode == 0, result.stderr
    assert "compare" in result.stdout
    assert "--help" in result.stdout
    assert "numpy imported" not in result.stderr


def test_compare_help_and_usage_errors_skip_numpy(tmp_path: Path) -> None:
    blocker = tmp_path / "numpy.py"
    blocker.write_text("raise RuntimeError('numpy imported during usage')\n", encoding="utf-8")
    blocked = f"{tmp_path}{os.pathsep}{_REPO / 'src'}"
    help_result = _run_qa_ts(["compare", "--help"], blocked)
    assert help_result.returncode == 0, help_result.stderr
    assert "--seed" in help_result.stdout
    missing = _run_qa_ts(["compare"], blocked)
    assert missing.returncode == 2
    assert "numpy imported" not in missing.stderr


def test_compare_cli_delegates_to_sweep(monkeypatch, tmp_path: Path) -> None:
    import ops.qa_ts_lib as lib
    from ops.qa_ts import main

    seen: dict = {}

    def fake_sweep(kind, sizes, seed, out_dir):
        seen["kind"] = kind
        seen["sizes"] = list(sizes)
        seen["seed"] = seed
        seen["out"] = out_dir
        return []

    monkeypatch.setattr(lib, "run_sweep", fake_sweep)
    main(["compare", "--function", "arma", "--seed", "3", "--out", str(tmp_path)])
    assert seen["kind"] == "arma"
    assert seen["sizes"] == [100, 300, 1000, 3000, 10000, 30000, 100000]
    assert seen["seed"] == 3
    assert seen["out"] == str(tmp_path)
