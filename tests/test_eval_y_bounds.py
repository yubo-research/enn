from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from evals import eval_y_bounds as mod

_EVAL_RE = re.compile(
    r"EVAL: model = (unbounded|y_bounds_\([^)]+\)) "
    r"SMALLER\(rmse\) = \d+\.\d{4} SMALLER\(mae\) = \d+\.\d{4} "
    r"SMALLER\(nll\) = -?\d+\.\d{4} "
    r"SMALLER\(frac_nonpos_mu\) = \d+\.\d{4} "
    r"SMALLER\(frac_nonpos_samples\) = \d+\.\d{4} "
    r"SMALLER\(frac_oob_samples\) = \d+\.\d{4}"
)


def test_evaluate_calls_run_y_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_y_bounds(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(mod, "run_y_bounds", fake_run_y_bounds)
    mod.evaluate()
    assert calls == [{}]


def test_evaluate_prints_constraint_fraction_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod.evaluate()
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[0].startswith("num_train=")
    assert len(lines) == 3
    assert _EVAL_RE.fullmatch(lines[1])
    assert _EVAL_RE.fullmatch(lines[2])
    assert "model = unbounded" in lines[1]
    assert "model = y_bounds_(0,inf)" in lines[2]
    assert "frac_nonpos_mu" in lines[1]
    assert "frac_oob_samples" in lines[2]


def test_evaluator_run_y_bounds() -> None:
    from ops.evaluator import cli

    result = CliRunner().invoke(cli, ["run", "y_bounds"])
    assert result.exit_code == 0, result.output
    assert "model = unbounded" in result.output
    assert "model = y_bounds_(0,inf)" in result.output
    assert "SMALLER(frac_oob_samples)" in result.output
