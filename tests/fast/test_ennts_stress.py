from __future__ import annotations

import re

from evals.short import eval_ennts_hnsw, eval_ennts_horizon, eval_ennts_metric

_CORE = (
    r"EVAL: aspect = {aspect} seed = \d+ n = \d+ "
    r"LARGER\(loglik\) = -?\d+\.\d{{4}} "
    r"SMALLER\(nrmse\) = \d+\.\d{{4}} "
    r"SMALLER\(add_s\) = \d+\.\d{{6}} "
    r"SMALLER\(query_s\) = \d+\.\d{{6}} "
)


def _eval_lines(capsys):
    out = capsys.readouterr().out
    return [line for line in out.splitlines() if line.startswith("EVAL:")]


def test_hnsw_eval_reports_tree_scale_and_rolling_fit(capsys) -> None:
    eval_ennts_hnsw.evaluate()
    pattern = re.compile(_CORE.format(aspect="hnsw") + r"windows = (?P<windows>\d+)")
    lines = _eval_lines(capsys)
    assert len(lines) == 2
    for line in lines:
        match = pattern.fullmatch(line)
        assert match is not None
        assert int(match.group("windows")) > 2048
        assert "nan" not in line


def test_horizon_eval_reports_delay_and_rolling_fit(capsys) -> None:
    eval_ennts_horizon.evaluate()
    pattern = re.compile(_CORE.format(aspect="horizon") + r"horizon = 12 cadence = 3")
    lines = _eval_lines(capsys)
    assert len(lines) == 2
    for line in lines:
        assert pattern.fullmatch(line)
        assert "nan" not in line


def test_metric_eval_learns_a_scale_while_rolling(capsys) -> None:
    eval_ennts_metric.evaluate()
    pattern = re.compile(
        _CORE.format(aspect="metric") + r"learned = (?P<learned>[01]) morph = \d+"
    )
    lines = _eval_lines(capsys)
    assert len(lines) == 2
    for line in lines:
        match = pattern.fullmatch(line)
        assert match is not None
        assert match.group("learned") == "1"
        assert "nan" not in line
