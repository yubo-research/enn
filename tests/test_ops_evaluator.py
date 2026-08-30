from __future__ import annotations

from click.testing import CliRunner


def test_list_eval_names_includes_known_evals() -> None:
    from ops.evaluator import list_eval_names

    names = list_eval_names()
    assert "turbo_enn" in names
    assert "enn_flat" in names
    assert "flat_sphere_d10" in names
    assert "flat_sphere_d100" in names
    assert "flat_sphere_d1000" in names
    assert "fast_mem_sphere_d10" in names
    assert "fast_mem_sphere_d100" in names
    assert "fast_mem_sphere_d1000" in names
    assert "enn_fast_mem" in names
    assert "ts_fast_mem" in names
    assert "bpann_sphere_d10" in names
    assert "bpann_sphere_d100" in names
    assert "bpann_sphere_d1000" in names
    assert "y_bounds" in names
    assert "turbo_acq" in names
    assert "y_var_noise" in names
    assert names == sorted(names)
    assert all(not name.startswith("stress") for name in names)


def test_list_command_prints_names() -> None:
    from ops.evaluator import cli, list_eval_names

    result = CliRunner().invoke(cli, ["list"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert lines == list_eval_names()


def test_run_unknown_eval_fails() -> None:
    from ops.evaluator import cli

    result = CliRunner().invoke(cli, ["run", "does_not_exist_xyz"])
    assert result.exit_code != 0
    assert "missing eval module" in result.output


def test_run_invokes_evaluate(monkeypatch) -> None:
    from ops import evaluator

    called: list[str] = []

    def fake_load(name: str):
        def evaluate() -> None:
            called.append(name)

        return evaluate

    monkeypatch.setattr(evaluator, "load_evaluate", fake_load)
    result = CliRunner().invoke(evaluator.cli, ["run", "turbo_enn"])
    assert result.exit_code == 0, result.output
    assert called == ["turbo_enn"]


def test_ensure_repo_on_sys_path_adds_repo_root(monkeypatch) -> None:
    import sys

    from ops import evaluator

    root = str(evaluator.REPO_ROOT)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != root])
    assert root not in sys.path
    evaluator.ensure_repo_on_sys_path()
    assert sys.path[0] == root
    evaluator.ensure_repo_on_sys_path()
    assert sys.path.count(root) == 1


def test_load_evaluate_turbo_enn_without_preexisting_pythonpath(
    monkeypatch,
) -> None:
    import sys

    from ops import evaluator

    root = str(evaluator.REPO_ROOT)
    monkeypatch.setattr(
        sys,
        "path",
        [p for p in sys.path if p not in (root, str(evaluator.REPO_ROOT / "evals"))],
    )
    evaluate = evaluator.load_evaluate("turbo_enn")
    assert callable(evaluate)
