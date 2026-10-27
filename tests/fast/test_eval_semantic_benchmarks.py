"""The semantic benchmarks use the same discovery and call contract as other evals."""

import csv
import inspect
from pathlib import Path

from click.testing import CliRunner

from _reports.semantic_features import eval_lunar_benchmark
from enn import SemanticInputTransform
from ops.evaluate import cli, load_evaluate


def test_benchmark_entry_point_accepts_no_arguments():
    evaluate = load_evaluate("long/semantic_benchmark")
    inspect.signature(evaluate).bind()


def test_semantic_benchmark_runs_through_evaluator(monkeypatch, tmp_path):
    evaluate = load_evaluate("long/semantic_benchmark")
    calls = []

    def run_case(seed, dimensions):
        calls.append((seed, dimensions))
        return [{"seed": seed, "dimensions": dimensions, "nrmse": 0.5}]

    monkeypatch.setitem(evaluate.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(evaluate.__globals__, "run_case", run_case)
    monkeypatch.setitem(evaluate.__globals__, "summarize", lambda rows: rows)
    monkeypatch.setattr("ops.evaluate.load_evaluate", lambda name: evaluate)
    result = CliRunner().invoke(cli, ["run", "long/semantic_benchmark"])
    assert result.exit_code == 0, result.output
    assert calls == [(seed, d) for d in (50, 100, 200) for seed in range(5)]
    for filename in ("semantic_raw.csv", "semantic_summary.csv"):
        with (tmp_path / filename).open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 15
        assert rows[-1] == {"seed": "4", "dimensions": "200", "nrmse": "0.5"}


def test_lunar_benchmark_uses_moved_assets():
    evaluate = eval_lunar_benchmark.evaluate
    root = evaluate.__globals__["ROOT"]
    assert (
        root == Path(__file__).resolve().parents[2] / "_reports" / "semantic_features"
    )
    transform = SemanticInputTransform.load(root / "lunar_projection.npz")
    assert transform is not None
