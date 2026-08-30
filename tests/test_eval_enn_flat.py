from __future__ import annotations

import io
import os
from types import SimpleNamespace

import pytest

from evals import eval_enn_flat as mod
from evals import stress_eval as shared


class _FakePopen:
    def __init__(self, lines: list[str], rc: int) -> None:
        self.stdout = io.StringIO("".join(lines))
        self._rc = rc

    def __enter__(self) -> _FakePopen:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def wait(self) -> int:
        return self._rc


def test_emit_and_pythonpath(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = shared.pythonpath_env()
    assert env["PYTHONPATH"] == str(shared.REPO_ROOT / "src")

    monkeypatch.setenv("PYTHONPATH", "/prior")
    env2 = shared.pythonpath_env()
    assert env2["PYTHONPATH"].startswith(str(shared.REPO_ROOT / "src"))
    assert env2["PYTHONPATH"].endswith(f"{os.pathsep}/prior")

    shared.emit_eval_line("num_dim = 10 num_obs = 3")
    shared.emit_eval_line("n = 3 query_s = 0.1 segment_s = 0.2")
    out = capsys.readouterr().out
    assert "EVAL: num_dim" not in out
    assert "num_dim = 10 num_obs = 3" in out
    assert (
        "EVAL: n = 3 SMALLER(query_s) = 0.1 SMALLER(segment_s) = 0.2" in out
    )


def test_evaluate_streams_and_exit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[SimpleNamespace] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakePopen:
        calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return _FakePopen(
            ["num_dim = 10 num_obs = 1\n", "n = 1 query_s = 0.01 segment_s = 0.02\n"],
            0,
        )

    monkeypatch.setattr(shared.subprocess, "Popen", fake_popen)
    mod.evaluate()
    out = capsys.readouterr().out
    assert "EVAL: n = 1 SMALLER(query_s) = 0.01 SMALLER(segment_s) = 0.02" in out
    assert calls and str(shared.NUM_OBS) in calls[0].args[0]
    assert "flat" in calls[0].args[0]

    def fake_fail(*args: object, **kwargs: object) -> _FakePopen:
        return _FakePopen([], 7)

    monkeypatch.setattr(shared.subprocess, "Popen", fake_fail)
    with pytest.raises(SystemExit) as exc:
        mod.evaluate()
    assert exc.value.code == 7
