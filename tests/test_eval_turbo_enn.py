from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from evals import eval_turbo_enn as mod
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


def test_emit_turbo_enn_line(capsys: pytest.CaptureFixture[str]) -> None:
    holder: list[float | None] = [None]
    shared.emit_turbo_enn_line(
        "num_dim=10 num_obs=1000 num_ask=100 index_type=flat",
        y_best_holder=holder,
    )
    shared.emit_turbo_enn_line("  10 0.0100 0.0010 0.0090", y_best_holder=holder)
    shared.emit_turbo_enn_line("y_best=-1.2345", y_best_holder=holder)
    out = capsys.readouterr().out
    assert "num_dim=10 num_obs=1000" in out
    assert "10 0.0100" in out
    assert "y_best=" not in out
    assert holder[0] == pytest.approx(-1.2345)


def test_evaluate_streams_and_eval_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[SimpleNamespace] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakePopen:
        calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return _FakePopen(
            [
                "num_dim=10 num_obs=1000 num_ask=100 index_type=flat\n",
                "   1 0.0100 0.0010 0.0090\n",
                "1000 0.0200 0.0020 0.0180\n",
                "y_best=-0.5\n",
            ],
            0,
        )

    monkeypatch.setattr(shared.subprocess, "Popen", fake_popen)
    times = iter([1.0, 3.5, 10.0, 11.0])
    monkeypatch.setattr(shared.time, "perf_counter", lambda: next(times))
    mod.evaluate()
    out = capsys.readouterr().out
    assert calls
    cmd = calls[0].args[0]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("turbo-enn") + 1 :] == ["flat", "1000", "100"]
    assert "num_dim=10 num_obs=1000" in out
    assert "y_best=" not in out
    assert "EVAL: SMALLER(time_seconds) = 2.5 LARGER(y_best) = -0.5" in out

    def fake_fail(*args: object, **kwargs: object) -> _FakePopen:
        return _FakePopen([], 7)

    monkeypatch.setattr(shared.subprocess, "Popen", fake_fail)
    with pytest.raises(SystemExit) as exc:
        mod.evaluate()
    assert exc.value.code == 7


def test_evaluate_missing_y_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_popen(*args: object, **kwargs: object) -> _FakePopen:
        return _FakePopen(["num_dim=10\n"], 0)

    monkeypatch.setattr(shared.subprocess, "Popen", fake_popen)
    times = iter([0.0, 1.0])
    monkeypatch.setattr(shared.time, "perf_counter", lambda: next(times))
    with pytest.raises(SystemExit) as exc:
        mod.evaluate()
    assert "y_best" in str(exc.value)
