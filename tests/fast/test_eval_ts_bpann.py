from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from evals import eval_ts_bpann as mod
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


def test_evaluate_sweeps_bpann_and_exit(
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
        assert "--index-type=bpann_disk" in cmd
        assert "--work-dir" in cmd
        work_dir_idx = cmd.index("--work-dir")
        assert work_dir_idx + 1 < len(cmd)
        assert cmd[work_dir_idx + 1]
        assert (
            f"EVAL: n = {num_obs} method = posterior LARGER(avg_likelihood) = 0.1"
            in out
        )
        assert (
            f"EVAL: n = {num_obs} method = posterior_function_draw "
            "LARGER(avg_likelihood) = 0.3"
            in out
        )

    def fake_fail(*args: object, **kwargs: object) -> _FakePopen:
        return _FakePopen([], 7)

    monkeypatch.setattr(shared.subprocess, "Popen", fake_fail)
    with pytest.raises(SystemExit) as exc:
        mod.evaluate()
    assert exc.value.code == 7
