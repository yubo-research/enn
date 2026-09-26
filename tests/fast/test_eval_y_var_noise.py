from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from evals import eval_y_var_noise as mod

_MODEL_RE = re.compile(
    r"EVAL: model = (matched|none|wrong) "
    r"SMALLER\(nll\) = -?\d+\.\d{4} "
    r"LARGER\(calib_1se\) = \d+\.\d{4} "
    r"mean_se = \d+\.\d{4} "
    r"mean_se_ale = \d+\.\d{4}"
)
_SWEEP_RE = re.compile(
    r"EVAL: yvar_scale = \d+\.\d{4} LARGER\(mean_se_ale\) = \d+\.\d{4}"
)


def test_evaluate_calls_run_y_var_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_y_var_noise(**kwargs: object) -> tuple[list[object], list[object]]:
        calls.append(kwargs)
        return [], []

    monkeypatch.setattr(mod, "run_y_var_noise", fake_run_y_var_noise)
    mod.evaluate()
    assert calls == [{}]


def test_evaluate_prints_y_var_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod.evaluate()
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[0].startswith("num_train=")
    assert _MODEL_RE.fullmatch(lines[1])
    assert _MODEL_RE.fullmatch(lines[2])
    assert _MODEL_RE.fullmatch(lines[3])
    assert "model = matched" in lines[1]
    assert "model = none" in lines[2]
    assert "model = wrong" in lines[3]
    assert len(lines) >= 5
    for line in lines[4:]:
        assert _SWEEP_RE.fullmatch(line), line


def test_evaluator_run_y_var_noise() -> None:
    from ops.evaluator import cli

    result = CliRunner().invoke(cli, ["run", "y_var_noise"])
    assert result.exit_code == 0, result.output
    assert "model = matched" in result.output
    assert "yvar_scale =" in result.output
    assert "SMALLER(nll)" in result.output
