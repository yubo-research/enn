from __future__ import annotations

from click.testing import CliRunner


def test_list_eval_names_includes_known_evals() -> None:
    from ops.evaluate import list_eval_names

    names = list_eval_names()
    assert names == [
        "long/prediction",
        "long/scaling",
        "long/semantic_benchmark",
        "long/semantic_outputs",
        "short/optimization",
        "short/prediction",
    ]
    assert names == sorted(names)
    assert all("/stress" not in name for name in names)
    assert all(not name.rsplit("/", 1)[-1].startswith("stress") for name in names)


def test_list_command_prints_names() -> None:
    from ops.evaluate import cli, list_eval_names

    result = CliRunner().invoke(cli, ["list"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert lines == list_eval_names()


def test_list_command_prefix_filters() -> None:
    from ops.evaluate import cli, list_eval_names

    for prefix in ("short/", "sh", "long/", "lo"):
        result = CliRunner().invoke(cli, ["list", prefix])
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.splitlines() if line.strip()]
        assert lines == list_eval_names(prefix)
    short_only = list_eval_names("short/")
    assert short_only
    assert all(name.startswith("short/") for name in short_only)
    assert list_eval_names("sh") == short_only


def test_run_unknown_eval_fails() -> None:
    from ops.evaluate import cli

    result = CliRunner().invoke(cli, ["run", "does_not_exist_xyz"])
    assert result.exit_code != 0
    assert "no evals match" in result.output


def test_run_invokes_evaluate(monkeypatch) -> None:
    from ops import evaluate

    called: list[str] = []

    def fake_load(name: str):
        def evaluate_fn() -> None:
            called.append(name)

        return evaluate_fn

    monkeypatch.setattr(evaluate, "load_evaluate", fake_load)
    result = CliRunner().invoke(evaluate.cli, ["run", "short/optimization"])
    assert result.exit_code == 0, result.output
    assert called == ["short/optimization"]


def test_run_prefix_runs_all_selected(monkeypatch) -> None:
    from ops import evaluate

    called: list[str] = []

    def fake_load(name: str):
        def evaluate_fn() -> None:
            called.append(name)

        return evaluate_fn

    monkeypatch.setattr(evaluate, "load_evaluate", fake_load)
    for prefix in ("short/", "sh"):
        called.clear()
        result = CliRunner().invoke(evaluate.cli, ["run", prefix])
        assert result.exit_code == 0, result.output
        assert called == evaluate.list_eval_names(prefix)
        assert called
        assert all(name.startswith("short/") for name in called)

    called.clear()
    result = CliRunner().invoke(evaluate.cli, ["run", "short/prediction"])
    assert result.exit_code == 0, result.output
    assert called == evaluate.list_eval_names("short/prediction")
    assert len(called) == 1


def test_ensure_repo_on_sys_path_adds_repo_and_sources(monkeypatch) -> None:
    import sys

    from ops import evaluate

    roots = [str(evaluate.REPO_ROOT / "src"), str(evaluate.REPO_ROOT)]
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in roots])
    evaluate.ensure_repo_on_sys_path()
    assert sys.path[:2] == roots
    evaluate.ensure_repo_on_sys_path()
    assert all(sys.path.count(root) == 1 for root in roots)


def test_load_evaluate_optimization_without_preexisting_pythonpath(
    monkeypatch,
) -> None:
    import sys

    from ops import evaluate

    root = str(evaluate.REPO_ROOT)
    monkeypatch.setattr(
        sys,
        "path",
        [p for p in sys.path if p not in (root, str(evaluate.REPO_ROOT / "src"))],
    )
    evaluate_fn = evaluate.load_evaluate("short/optimization")
    assert callable(evaluate_fn)
