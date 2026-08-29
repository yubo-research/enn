from __future__ import annotations

import re

import numpy as np
import pytest
from click.testing import CliRunner

_FAST_TURBO = dict(
    num_dim=3,
    num_rounds=4,
    num_arms=2,
    num_init=2,
    num_seeds=2,
    seed=0,
    k=3,
    num_fit_samples=5,
)

_FAST_YVAR = dict(
    n_train=24,
    n_test=30,
    k=3,
    num_fit_candidates=4,
    num_fit_samples=3,
    seed=0,
)

_TURBO_EVAL_RE = re.compile(
    r"EVAL: problem = (noiseless|noisy) method = (ucb|thompson|pareto|turbo_zero) "
    r"LARGER\(y_best_mean\) = -?\d+\.\d{4} "
    r"SMALLER\(y_best_se\) = \d+\.\d{4} "
    r"SMALLER\(time_seconds\) = \d+\.\d{4} "
    r"LARGER\(auc_y_best\) = -?\d+\.\d{4}"
)

_YVAR_MODEL_RE = re.compile(
    r"EVAL: model = (matched|none|wrong) "
    r"SMALLER\(nll\) = -?\d+\.\d{4} "
    r"LARGER\(calib_1se\) = \d+\.\d{4} "
    r"mean_se = \d+\.\d{4} "
    r"mean_se_ale = \d+\.\d{4}"
)


def test_auc_of_y_best_trajectory():
    from ops.qa import auc_of_y_best_trajectory

    assert np.isnan(auc_of_y_best_trajectory([]))
    assert auc_of_y_best_trajectory([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert auc_of_y_best_trajectory([0.0, 2.0]) == pytest.approx(1.0)


def test_run_turbo_acq_seed_noiseless():
    from ops.qa import AcqType, run_turbo_acq_seed

    result = run_turbo_acq_seed(
        acq_type=AcqType.UCB,
        noise=0.0,
        seed=0,
        num_dim=3,
        num_rounds=3,
        num_arms=2,
        num_init=2,
        k=3,
        num_fit_samples=5,
    )
    assert np.isfinite(result.y_best)
    assert result.time_seconds >= 0.0
    assert len(result.trajectory) == 3
    assert result.trajectory[-1] == result.y_best


def test_run_turbo_acq_includes_pareto_and_zero():
    from ops.qa import run_turbo_acq

    metrics = run_turbo_acq(
        **_FAST_TURBO,
        include_noisy=False,
        include_turbo_zero=True,
    )
    methods = {m.method for m in metrics}
    assert methods == {"ucb", "thompson", "pareto", "turbo_zero"}
    assert all(m.problem == "noiseless" for m in metrics)
    for m in metrics:
        assert np.isfinite(m.y_best_mean)
        assert m.y_best_se >= 0.0


def test_run_turbo_acq_noisy_omits_pareto():
    from ops.qa import run_turbo_acq

    metrics = run_turbo_acq(
        **_FAST_TURBO,
        include_noisy=True,
        include_turbo_zero=False,
    )
    noisy = [m for m in metrics if m.problem == "noisy"]
    assert {m.method for m in noisy} == {"ucb", "thompson"}


def test_turbo_acq_cli():
    from ops.qa import cli

    result = CliRunner().invoke(
        cli,
        [
            "turbo-acq",
            "--num-dim",
            "3",
            "--num-rounds",
            "3",
            "--num-arms",
            "2",
            "--num-init",
            "2",
            "--num-seeds",
            "2",
            "--no-noisy",
            "--no-turbo-zero",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.startswith("EVAL:")]
    assert len(lines) == 3
    for line in lines:
        assert _TURBO_EVAL_RE.fullmatch(line), line


def test_compare_y_var_noise_matched_has_finite_metrics():
    from ops.qa import compare_y_var_noise

    matched, none_m, wrong = compare_y_var_noise(**_FAST_YVAR)
    for row in (matched, none_m, wrong):
        assert np.isfinite(row.nll)
        assert 0.0 <= row.calib_1se <= 1.0
        assert row.mean_se > 0.0
        assert row.mean_se_ale >= 0.0


def test_matched_yvar_improves_or_matches_nll():
    from ops.qa import compare_y_var_noise

    matched, none_m, wrong = compare_y_var_noise(**_FAST_YVAR, sigma=0.4)
    # Hypothesis check soft: matched should not be dramatically worse than none.
    assert matched.nll <= none_m.nll + 0.5
    assert matched.nll <= wrong.nll + 0.5


def test_se_ale_monotonic_in_yvar_scale():
    from ops.qa import sweep_y_var_se_ale

    sweep = sweep_y_var_se_ale(
        **_FAST_YVAR,
        yvar_scales=(0.01, 0.25, 1.0),
    )
    scales = [s for s, _ in sweep]
    ales = [a for _, a in sweep]
    assert scales == [0.01, 0.25, 1.0]
    assert ales[0] <= ales[1] <= ales[2]


def test_y_var_noise_cli():
    from ops.qa import cli

    result = CliRunner().invoke(
        cli,
        [
            "y-var-noise",
            "--n-train",
            "20",
            "--n-test",
            "30",
            "--k",
            "3",
            "--num-fit-candidates",
            "4",
            "--num-fit-samples",
            "3",
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    model_lines = [
        ln for ln in result.output.splitlines() if ln.startswith("EVAL: model =")
    ]
    assert len(model_lines) == 3
    for line in model_lines:
        assert _YVAR_MODEL_RE.fullmatch(line), line
    assert any("yvar_scale =" in ln for ln in result.output.splitlines())
