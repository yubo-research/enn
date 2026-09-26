from __future__ import annotations

from click.testing import CliRunner


def test_list_eval_names_includes_known_evals() -> None:
    from ops.evaluate import list_eval_names

    names = list_eval_names()
    assert "short/turbo_enn" in names
    assert "short/enn_flat" in names
    assert "short/flat_sphere_d10" in names
    assert "short/flat_sphere_d100" in names
    assert "short/flat_sphere_d1000" in names
    assert "short/bpann_sphere_d10" in names
    assert "short/bpann_sphere_d100" in names
    assert "short/bpann_sphere_d1000" in names
    assert "short/bpann_persist_stability" in names
    assert "short/y_bounds" in names
    assert "short/turbo_acq" in names
    assert "short/y_var_noise" in names
    assert "long/enn_bpann_disk" in names
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
    assert "missing eval module" in result.output


def test_run_invokes_evaluate(monkeypatch) -> None:
    from ops import evaluate

    called: list[str] = []

    def fake_load(name: str):
        def evaluate_fn() -> None:
            called.append(name)

        return evaluate_fn

    monkeypatch.setattr(evaluate, "load_evaluate", fake_load)
    result = CliRunner().invoke(evaluate.cli, ["run", "short/turbo_enn"])
    assert result.exit_code == 0, result.output
    assert called == ["short/turbo_enn"]


def test_ensure_repo_on_sys_path_adds_repo_root(monkeypatch) -> None:
    import sys

    from ops import evaluate

    root = str(evaluate.REPO_ROOT)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != root])
    assert root not in sys.path
    evaluate.ensure_repo_on_sys_path()
    assert sys.path[0] == root
    evaluate.ensure_repo_on_sys_path()
    assert sys.path.count(root) == 1


def test_load_evaluate_turbo_enn_without_preexisting_pythonpath(
    monkeypatch,
) -> None:
    import sys

    from ops import evaluate

    root = str(evaluate.REPO_ROOT)
    monkeypatch.setattr(
        sys,
        "path",
        [p for p in sys.path if p not in (root, str(evaluate.REPO_ROOT / "evals"))],
    )
    evaluate_fn = evaluate.load_evaluate("short/turbo_enn")
    assert callable(evaluate_fn)
