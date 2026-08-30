from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from evals import eval_turbo_acq as mod

_EVAL_RE = re.compile(
    r"EVAL: problem = (noiseless|noisy) method = (ucb|thompson|pareto|turbo_zero) "
    r"LARGER\(y_best_mean\) = -?\d+\.\d{4} "
    r"SMALLER\(y_best_se\) = \d+\.\d{4} "
    r"SMALLER\(time_seconds\) = \d+\.\d{4} "
    r"LARGER\(auc_y_best\) = -?\d+\.\d{4}"
)


def test_evaluate_calls_run_turbo_acq(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_turbo_acq(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(mod, "run_turbo_acq", fake_run_turbo_acq)
    mod.evaluate()
    assert calls == [{}]


def test_evaluate_prints_acq_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ops.qa import TurboAcqMetrics, echo_turbo_acq_metrics

    def fake_run_turbo_acq(**kwargs: object) -> list[TurboAcqMetrics]:
        metrics = [
            TurboAcqMetrics("noiseless", "ucb", -1.0, 0.1, 0.5, -1.2),
            TurboAcqMetrics("noiseless", "thompson", -1.1, 0.2, 0.4, -1.3),
            TurboAcqMetrics("noiseless", "pareto", -0.9, 0.05, 0.3, -1.0),
            TurboAcqMetrics("noiseless", "turbo_zero", -2.0, 0.3, 0.01, -2.1),
            TurboAcqMetrics("noisy", "ucb", -1.5, 0.15, 0.6, -1.6),
            TurboAcqMetrics("noisy", "thompson", -1.6, 0.25, 0.55, -1.7),
        ]
        print(
            "num_dim=5 num_rounds=15 num_arms=5 num_seeds=3 "
            "include_noisy=True include_turbo_zero=True"
        )
        for row in metrics:
            echo_turbo_acq_metrics(row)
        return metrics

    monkeypatch.setattr(mod, "run_turbo_acq", fake_run_turbo_acq)
    mod.evaluate()
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[0].startswith("num_dim=")
    assert len(lines) == 7
    for line in lines[1:]:
        assert _EVAL_RE.fullmatch(line), line
    assert "method = pareto" in lines[3]
    assert "problem = noisy" in lines[5]


def test_evaluator_run_turbo_acq(monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import evaluator
    from ops.evaluator import cli

    def fake_run(**kwargs: object) -> list[object]:
        print(
            "EVAL: problem = noiseless method = ucb "
            "LARGER(y_best_mean) = -1.0000 "
            "SMALLER(y_best_se) = 0.1000 "
            "SMALLER(time_seconds) = 0.5000 "
            "LARGER(auc_y_best) = -1.2000"
        )
        return []

    monkeypatch.setenv(evaluator._EVAL_INLINE_ENV, "1")
    monkeypatch.setattr(evaluator, "prepare_eval_process", lambda **_: None)
    monkeypatch.setattr(mod, "run_turbo_acq", fake_run)
    # load_evaluate imports a fresh module object; patch the runner it will bind.
    monkeypatch.setattr("ops.qa.run_turbo_acq", fake_run)
    result = CliRunner().invoke(cli, ["run", "turbo_acq"])
    assert result.exit_code == 0, result.output
    assert "method = ucb" in result.output
    assert "LARGER(y_best_mean)" in result.output
