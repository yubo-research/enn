from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from scripts.compare_ennbo_versions import (
    BASELINE_LABEL,
    CURRENT_LABEL,
    _build_parser,
    _compare_summary,
    _env_for_version,
    _format_row,
    _run_worker_subprocess,
    _worker_main,
    main,
)
from scripts.compare_ennbo_versions_common import (
    BenchmarkResult,
    OPTIMIZER_NAMES,
    PROBLEMS,
    apply_quick_overrides,
    build_optimizer_config,
    compute_hypervolume,
    experiment_combos,
    run_benchmark,
    separable_unimodal_objective,
)


def test_experiment_grid_has_nine_cells():
    combos = experiment_combos()
    assert len(combos) == 9
    assert len({c[0] for c in combos}) == 3


def test_optimizer_names_match_plan():
    assert OPTIMIZER_NAMES == ("turbo_enn", "turbo_one", "morbo")


def test_separable_unimodal_objective_peak():
    x = np.array([[120.0, 0.91]])
    y = separable_unimodal_objective(x)
    assert y.shape == (1, 2)
    assert float(y[0, 0]) >= 499_000.0
    assert float(y[0, 1]) >= 11.0


def test_compute_hypervolume_nonempty():
    y = np.array([[1.0, 2.0], [3.0, 1.5], [0.5, 3.0]])
    hv = compute_hypervolume(y, np.array([0.0, 0.0]))
    assert hv > 0.0


def test_build_optimizer_config_morbo_requires_multi_objective():
    single = PROBLEMS["ackley_30d"]
    with pytest.raises(ValueError, match="num_metrics"):
        build_optimizer_config("morbo", single)


def test_apply_quick_overrides_reduces_iterations():
    quick = apply_quick_overrides(PROBLEMS)
    assert quick["ackley_30d"].num_iterations < PROBLEMS["ackley_30d"].num_iterations


def test_run_benchmark_turbo_enn_ackley_quick():
    problem = apply_quick_overrides(PROBLEMS)["ackley_30d"]
    result = run_benchmark(
        optimizer="turbo_enn",
        problem=problem,
        version_label="current",
    )
    assert result.quality_metric == "best_y"
    assert np.isfinite(result.quality)
    assert result.num_evals == problem.num_iterations * problem.num_arms


def test_run_benchmark_morbo_separable_unimodal_quick():
    problem = apply_quick_overrides(PROBLEMS)["separable_unimodal"]
    result = run_benchmark(
        optimizer="morbo",
        problem=problem,
        version_label="current",
    )
    assert result.quality_metric == "hypervolume"
    assert np.isfinite(result.quality)


def test_compare_summary_metamorphic_quality_shift():
    current = BenchmarkResult(
        optimizer="turbo_enn",
        problem="ackley_30d",
        version_label=CURRENT_LABEL,
        quality=2.0,
        quality_metric="best_y",
        wall_seconds=1.0,
        ask_seconds=0.5,
        num_evals=20,
        seed=18,
    )
    baseline = BenchmarkResult(
        **{**asdict(current), "version_label": BASELINE_LABEL, "quality": 1.0}
    )
    summary = _compare_summary(current, baseline)
    assert summary["quality_delta"] == pytest.approx(1.0)
    assert summary["wall_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_compare_summary_fuzz_independent(seed: int):
    rng = np.random.default_rng(seed)
    cur_q = float(rng.normal())
    base_q = float(rng.normal())
    current = BenchmarkResult(
        optimizer="turbo_one",
        problem="ackley_30d",
        version_label=CURRENT_LABEL,
        quality=cur_q,
        quality_metric="best_y",
        wall_seconds=1.0,
        ask_seconds=0.5,
        num_evals=20,
        seed=seed,
    )
    baseline = BenchmarkResult(
        **{**asdict(current), "version_label": BASELINE_LABEL, "quality": base_q}
    )
    summary = _compare_summary(current, baseline)
    assert summary["quality_delta"] == pytest.approx(cur_q - base_q)
    print(f"compare_summary fuzz seed={seed}")


def test_env_for_version_sets_pythonpath_for_current():
    env = _env_for_version(CURRENT_LABEL)
    assert str(env["PYTHONPATH"]).split(":")[0].endswith("/src")


def test_env_for_version_strips_src_for_baseline():
    env = _env_for_version(BASELINE_LABEL)
    assert "/src" not in env.get("PYTHONPATH", "")


def test_format_row_contains_labels():
    row = _format_row(
        BenchmarkResult(
            optimizer="morbo",
            problem="separable_unimodal",
            version_label=CURRENT_LABEL,
            quality=1.0,
            quality_metric="hypervolume",
            wall_seconds=2.0,
            ask_seconds=1.0,
            num_evals=40,
            seed=42,
        )
    )
    assert "morbo" in row
    assert "separable_unimodal" in row


def test_run_worker_subprocess_dispatch(monkeypatch):
    from scripts import compare_ennbo_versions as module

    expected = BenchmarkResult(
        optimizer="turbo_enn",
        problem="ackley_30d",
        version_label=CURRENT_LABEL,
        quality=-1.0,
        quality_metric="best_y",
        wall_seconds=0.1,
        ask_seconds=0.05,
        num_evals=20,
        seed=18,
    )

    class Proc:
        returncode = 0
        stdout = json.dumps(expected.to_dict())
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: Proc())
    result = _run_worker_subprocess(
        optimizer="turbo_enn",
        problem="ackley_30d",
        version_label=CURRENT_LABEL,
        quick=True,
    )
    assert result.quality == pytest.approx(-1.0)


def test_main_quick_smoke(monkeypatch, tmp_path):
    from scripts import compare_ennbo_versions as module

    def fake_worker(*, optimizer, problem, version_label, quick):
        return BenchmarkResult(
            optimizer=optimizer,
            problem=problem,
            version_label=version_label,
            quality=0.5,
            quality_metric="best_y",
            wall_seconds=0.1,
            ask_seconds=0.05,
            num_evals=10,
            seed=18,
        )

    monkeypatch.setattr(module, "_run_worker_subprocess", fake_worker)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    rc = main(["--quick"])
    assert rc == 0
    report = json.loads((tmp_path / "compare_ennbo_versions_report.json").read_text())
    assert report["baseline_version"] == "0.3.6"
    assert len(report["comparisons"]) == 9


def test_worker_main_and_parser(monkeypatch, capsys):
    from scripts import compare_ennbo_versions as module

    monkeypatch.setattr(
        module,
        "run_benchmark",
        lambda **kwargs: BenchmarkResult(
            optimizer=kwargs["optimizer"],
            problem=kwargs["problem"],
            version_label=kwargs["version_label"],
            quality=0.0,
            quality_metric="best_y",
            wall_seconds=0.1,
            ask_seconds=0.05,
            num_evals=10,
            seed=18,
        ),
    )
    parser = _build_parser()
    args = parser.parse_args(
        [
            "worker",
            "--optimizer",
            "turbo_enn",
            "--problem",
            "ackley_30d",
            "--version",
            "current",
            "--quick",
        ]
    )
    assert _worker_main(args) == 0
    assert json.loads(capsys.readouterr().out)["optimizer"] == "turbo_enn"


def test_experiment_combos_morbo_uses_three_multi_objective_problems():
    morbo_problems = [p for o, p in experiment_combos() if o == "morbo"]
    assert morbo_problems == [
        "double_ackley_30d",
        "separable_unimodal",
        "ackley_pair_30d",
    ]


def test_problem_spec_bounds_ackley():
    spec = PROBLEMS["ackley_30d"]
    bounds = spec.bounds()
    assert bounds.shape == (30, 2)


def test_problem_spec_make_objective_ackley_pair():
    spec = PROBLEMS["ackley_pair_30d"]
    objective = spec.make_objective()
    y = objective(np.zeros((2, 30)))
    assert y.shape == (2, 2)
