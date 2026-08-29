from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from evals import eval_enn_bpann_disk as mod
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
    assert "EVAL: n = 1" in out
    assert calls
    cmd = calls[0].args[0]
    assert "10000000" in cmd
    assert "bpann_disk" in cmd
    assert "--work-dir" in cmd
    work_dir_idx = cmd.index("--work-dir")
    assert work_dir_idx + 1 < len(cmd)
    assert cmd[work_dir_idx + 1]

    def fake_fail(*args: object, **kwargs: object) -> _FakePopen:
        return _FakePopen([], 7)

    monkeypatch.setattr(shared.subprocess, "Popen", fake_fail)
    with pytest.raises(SystemExit) as exc:
        mod.evaluate()
    assert exc.value.code == 7
