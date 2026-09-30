from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager

from click.testing import CliRunner


@asynccontextmanager
async def _null_context():
    yield None


def _collect(names, call):
    from ops.evaluate_modal import run_all

    lines: list[str] = []
    failed = asyncio.run(run_all(names, lines.append, call, _null_context()))
    return lines, failed


def test_eval_command_runs_exact_eval_id() -> None:
    from ops.evaluate_modal import eval_command

    cmd = eval_command("short/bpann_sphere_d10")
    assert cmd[0] == sys.executable
    assert cmd[-1] == "short/bpann_sphere_d10"
    assert "load_evaluate" in cmd[2]


def test_run_eval_captured_merges_stdout_and_stderr(monkeypatch) -> None:
    from ops import evaluate_modal

    code = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"
    monkeypatch.setattr(
        evaluate_modal, "eval_command", lambda _eval_id: [sys.executable, "-c", code]
    )
    rc, output = evaluate_modal.run_eval_captured("short/x")
    assert rc == 3
    assert "out" in output
    assert "err" in output


def test_format_result_marks_status_and_frames_output() -> None:
    from ops.evaluate_modal import format_result

    ok = format_result("short/a", 0, "line1\nline2")
    assert ok.splitlines() == [
        "===== short/a (ok) =====",
        "line1",
        "line2",
        "===== end short/a =====",
    ]
    bad = format_result("short/b", 2, "")
    assert bad.splitlines() == ["===== short/b (FAILED rc=2) =====", "===== end short/b ====="]


def test_run_all_prints_each_eval_atomically_in_completion_order() -> None:
    delays = {"short/slow": 0.05, "short/fast": 0.0}

    async def call(eval_id: str) -> tuple[int, str]:
        await asyncio.sleep(delays[eval_id])
        return 0, f"{eval_id} line1\n{eval_id} line2\n"

    lines, failed = _collect(["short/slow", "short/fast"], call)
    assert failed == []
    assert len(lines) == 2
    assert lines[0].startswith("===== short/fast (ok)")
    assert "short/fast line1\nshort/fast line2\n" in lines[0]
    assert "short/slow" not in lines[0]
    assert lines[1].startswith("===== short/slow (ok)")


def test_run_all_reports_failures_and_exceptions() -> None:
    async def call(eval_id: str) -> tuple[int, str]:
        if eval_id == "short/boom":
            raise RuntimeError("container died")
        return (1, "bad\n") if eval_id == "short/bad" else (0, "good\n")

    lines, failed = _collect(["short/good", "short/bad", "short/boom"], call)
    assert sorted(failed) == ["short/bad", "short/boom"]
    boom = next(line for line in lines if "short/boom" in line)
    assert "RuntimeError: container died" in boom


def test_run_model_command_and_alias(monkeypatch) -> None:
    from ops import evaluate, evaluate_modal

    seen: list[list[str]] = []

    def fake_run_on_modal(names, echo):
        seen.append(list(names))
        echo("fake output")
        return []

    monkeypatch.setattr(evaluate_modal, "run_on_modal", fake_run_on_modal)
    for command in ("run-model", "rm"):
        result = CliRunner().invoke(evaluate.cli, [command, "short/y_"])
        assert result.exit_code == 0, result.output
        assert "fake output" in result.output
    assert seen == [evaluate.list_eval_names("short/y_")] * 2


def test_run_model_command_fails_when_an_eval_fails(monkeypatch) -> None:
    from ops import evaluate, evaluate_modal

    monkeypatch.setattr(evaluate_modal, "run_on_modal", lambda names, echo: names[:1])
    result = CliRunner().invoke(evaluate.cli, ["rm", "short/y_"])
    assert result.exit_code != 0
    assert "evals failed: short/y_bounds" in result.output


def test_run_model_unknown_prefix_fails() -> None:
    from ops import evaluate

    result = CliRunner().invoke(evaluate.cli, ["rm", "does_not_exist_xyz"])
    assert result.exit_code != 0
    assert "no evals match" in result.output
