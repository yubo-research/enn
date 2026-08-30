from __future__ import annotations

import os
import resource

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
    assert "missing eval module" in result.stderr or "missing eval module" in result.output


def test_run_invokes_evaluate(monkeypatch) -> None:
    from ops import evaluator

    called: list[str] = []

    def fake_load(name: str):
        def evaluate() -> None:
            called.append(name)

        return evaluate

    monkeypatch.setenv(evaluator._EVAL_INLINE_ENV, "1")
    monkeypatch.setattr(evaluator, "load_evaluate", fake_load)
    monkeypatch.setattr(evaluator, "prepare_eval_process", lambda **_: None)
    result = CliRunner().invoke(evaluator.cli, ["run", "turbo_enn"])
    assert result.exit_code == 0, result.output
    assert called == ["turbo_enn"]


def test_apply_eval_thread_limits_sets_env(monkeypatch) -> None:
    from ops import evaluator

    for key in evaluator._THREAD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    evaluator.apply_eval_thread_limits(1)
    for key in evaluator._THREAD_ENV_KEYS:
        assert os.environ[key] == "1"


def test_apply_eval_memory_limit_sets_rlimit_as(monkeypatch) -> None:
    from ops import evaluator

    calls: list[tuple[int, tuple[int, int]]] = []

    monkeypatch.setattr(
        resource,
        "getrlimit",
        lambda _which: (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
    )

    def fake_setrlimit(which: int, limits: tuple[int, int]) -> None:
        calls.append((which, limits))

    monkeypatch.setattr(resource, "setrlimit", fake_setrlimit)
    limit = 3 * 1024**3
    evaluator.apply_eval_memory_limit(limit)
    assert calls == [(resource.RLIMIT_AS, (limit, limit))]


def test_default_eval_memory_limit_is_3gib() -> None:
    from ops.evaluator import DEFAULT_EVAL_MEMORY_LIMIT_BYTES

    assert DEFAULT_EVAL_MEMORY_LIMIT_BYTES == 3 * 1024**3


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
