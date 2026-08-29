from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from evals import eval_ts as mod
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


def test_emit_eval_ts_line(capsys: pytest.CaptureFixture[str]) -> None:
    shared.emit_eval_ts_line(
        "num_dim=10 num_obs=100 num_test=100 seed=0",
        100,
    )
    shared.emit_eval_ts_line(
        "posterior avg_likelihood=0.8 argmin_rms=0.7 argmin_hit_rate=0.0100 eval_s=0.01",
        100,
    )
    shared.emit_eval_ts_line(
        "posterior_function_draw avg_likelihood=0.9 argmin_rms=0.6 "
        "argmin_hit_rate=0.0200 eval_s=0.02",
        1000,
    )
    out = capsys.readouterr().out
    assert "EVAL: " not in out.splitlines()[0]
    assert "num_dim=10 num_obs=100" in out
    assert (
        "EVAL: 100 posterior avg_likelihood=0.8 argmin_rms=0.7 "
        "argmin_hit_rate=0.0100 eval_s=0.01"
    ) in out
    assert (
        "EVAL: 1000 posterior_function_draw avg_likelihood=0.9 argmin_rms=0.6 "
        "argmin_hit_rate=0.0200 eval_s=0.02"
    ) in out


def test_evaluate_sweeps_and_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[SimpleNamespace] = []

    def fake_popen(*args: object, **kwargs: object) -> _FakePopen:
        calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        cmd = args[0]
        assert isinstance(cmd, list)
        num_obs = cmd[cmd.index("draw") + 1]
        return _FakePopen(
            [
                f"num_dim=10 num_obs={num_obs} num_test=100 seed=0\n",
                "posterior avg_likelihood=0.1 argmin_rms=0.2 "
                "argmin_hit_rate=0.0300 eval_s=0.01\n",
                "posterior_function_draw avg_likelihood=0.3 argmin_rms=0.4 "
                "argmin_hit_rate=0.0400 eval_s=0.02\n",
            ],
            0,
        )

    monkeypatch.setattr(shared.subprocess, "Popen", fake_popen)
    mod.evaluate()
    out = capsys.readouterr().out
    assert len(calls) == len(shared.TS_NUM_OBS_SWEEP)
    assert shared.TS_NUM_OBS_SWEEP == (100, 1_000, 10_000)
    for call, num_obs in zip(calls, shared.TS_NUM_OBS_SWEEP, strict=True):
        cmd = call.args[0]
        draw_i = cmd.index("draw")
        assert cmd[draw_i + 1] == str(num_obs)
        assert cmd[draw_i + 2] == "100"
        assert "--num-seeds=100" in cmd
        assert f"EVAL: {num_obs} posterior avg_likelihood=0.1" in out
        assert f"EVAL: {num_obs} posterior_function_draw avg_likelihood=0.3" in out

    def fake_fail(*args: object, **kwargs: object) -> _FakePopen:
        return _FakePopen([], 7)

    monkeypatch.setattr(shared.subprocess, "Popen", fake_fail)
    with pytest.raises(SystemExit) as exc:
        mod.evaluate()
    assert exc.value.code == 7
