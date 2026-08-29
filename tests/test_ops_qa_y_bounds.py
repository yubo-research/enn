from __future__ import annotations

import re

import numpy as np
import pytest
from click.testing import CliRunner

_FAST_KWARGS = dict(
    n_train=24,
    n_test=30,
    k=3,
    num_fit_candidates=4,
    num_fit_samples=3,
    num_draws=16,
    seed=0,
)

_EVAL_RE = re.compile(
    r"EVAL: model = (unbounded|y_bounds_\([^)]+\)) "
    r"SMALLER\(rmse\) = \d+\.\d{4} SMALLER\(mae\) = \d+\.\d{4} "
    r"SMALLER\(nll\) = -?\d+\.\d{4} "
    r"SMALLER\(frac_nonpos_mu\) = \d+\.\d{4} "
    r"SMALLER\(frac_nonpos_samples\) = \d+\.\d{4} "
    r"SMALLER\(frac_oob_samples\) = \d+\.\d{4}"
)


def _assert_finite_metrics(metrics) -> None:
    assert np.isfinite(metrics.rmse)
    assert np.isfinite(metrics.mae)
    assert np.isfinite(metrics.nll)


def _assert_bounded_support(bounded, lo: float, hi: float) -> None:
    assert bounded.frac_oob_samples == 0.0
    if lo == 0.0 and hi == np.inf:
        assert bounded.frac_nonpos_samples == 0.0


def _assert_matched_seed_reproducible(compare_fn, *args, **kwargs) -> None:
    first = compare_fn(*args, **kwargs)
    second = compare_fn(*args, **kwargs)
    for field in ("rmse", "mae", "nll"):
        assert getattr(first[0], field) == getattr(second[0], field)
        assert getattr(first[1], field) == getattr(second[1], field)


def test_make_positive_1d_xy_is_strictly_positive():
    from ops.qa import make_positive_1d_xy

    rng = np.random.default_rng(0)
    x, y = make_positive_1d_xy(32, rng)
    assert x.shape == (32, 1)
    assert y.shape == (32, 1)
    assert np.all(y > 0.0)


def test_make_bounded_1d_xy_respects_open_interval():
    from ops.qa import make_bounded_1d_xy

    rng = np.random.default_rng(1)
    cases = [
        (0.0, np.inf),
        (0.0, 1.0),
        (-np.inf, 0.0),
        (-1.0, 1.0),
        (-3.5, 7.25),
        (-np.inf, 4.0),
        (2.0, np.inf),
    ]
    for lo, hi in cases:
        _, y = make_bounded_1d_xy(40, rng, lo, hi, y_scale=3.0, y_center=-1.5)
        if np.isfinite(lo):
            assert np.all(y > lo), (lo, hi)
        if np.isfinite(hi):
            assert np.all(y < hi), (lo, hi)


def test_y_bounds_cli_prints_both_models():
    from ops.qa import cli

    result = CliRunner().invoke(
        cli,
        [
            "y-bounds",
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
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_train=20 num_test=30 k=3"
    assert len(lines) == 3
    assert _EVAL_RE.fullmatch(lines[1])
    assert _EVAL_RE.fullmatch(lines[2])
    assert "model = unbounded" in lines[1]
    assert "model = y_bounds_(0,inf)" in lines[2]


def test_compare_unbounded_vs_bounded_zero_inf():
    from ops.qa import compare_unbounded_vs_bounded

    unbounded, bounded = compare_unbounded_vs_bounded(0.0, np.inf, **_FAST_KWARGS)
    _assert_finite_metrics(unbounded)
    _assert_finite_metrics(bounded)
    _assert_bounded_support(bounded, 0.0, np.inf)
    _assert_matched_seed_reproducible(
        compare_unbounded_vs_bounded, 0.0, np.inf, **_FAST_KWARGS
    )


def test_compare_unbounded_vs_bounded_zero_one():
    from ops.qa import compare_unbounded_vs_bounded

    unbounded, bounded = compare_unbounded_vs_bounded(0.0, 1.0, **_FAST_KWARGS)
    _assert_finite_metrics(unbounded)
    _assert_finite_metrics(bounded)
    _assert_bounded_support(bounded, 0.0, 1.0)
    _assert_matched_seed_reproducible(
        compare_unbounded_vs_bounded, 0.0, 1.0, **_FAST_KWARGS
    )


def test_compare_unbounded_vs_bounded_neg_inf_zero():
    from ops.qa import compare_unbounded_vs_bounded

    unbounded, bounded = compare_unbounded_vs_bounded(-np.inf, 0.0, **_FAST_KWARGS)
    _assert_finite_metrics(unbounded)
    _assert_finite_metrics(bounded)
    _assert_bounded_support(bounded, -np.inf, 0.0)
    _assert_matched_seed_reproducible(
        compare_unbounded_vs_bounded, -np.inf, 0.0, **_FAST_KWARGS
    )


def test_compare_unbounded_vs_bounded_neg_one_one():
    from ops.qa import compare_unbounded_vs_bounded

    unbounded, bounded = compare_unbounded_vs_bounded(-1.0, 1.0, **_FAST_KWARGS)
    _assert_finite_metrics(unbounded)
    _assert_finite_metrics(bounded)
    _assert_bounded_support(bounded, -1.0, 1.0)
    _assert_matched_seed_reproducible(
        compare_unbounded_vs_bounded, -1.0, 1.0, **_FAST_KWARGS
    )


@pytest.mark.parametrize(
    "seed,y_scale,y_center",
    [
        (101, 0.2, -4.0),
        (202, 12.0, 0.0),
        (303, 45.0, 8.5),
    ],
)
def test_compare_unbounded_vs_bounded_random_finite(seed, y_scale, y_center):
    from ops.qa import compare_unbounded_vs_bounded

    rng = np.random.default_rng(seed)
    lo = float(rng.uniform(-12.0, 3.0))
    hi = lo + float(rng.uniform(0.5, 18.0))
    kwargs = {**_FAST_KWARGS, "seed": seed, "y_scale": y_scale, "y_center": y_center}
    unbounded, bounded = compare_unbounded_vs_bounded(lo, hi, **kwargs)
    _assert_finite_metrics(unbounded)
    _assert_finite_metrics(bounded)
    _assert_bounded_support(bounded, lo, hi)
    _assert_matched_seed_reproducible(compare_unbounded_vs_bounded, lo, hi, **kwargs)


@pytest.mark.parametrize(
    "seed,y_scale,y_center",
    [
        (401, 0.5, -2.0),
        (402, 8.0, 1.0),
        (403, 30.0, -6.0),
    ],
)
def test_compare_unbounded_vs_bounded_neg_inf_random_b(seed, y_scale, y_center):
    from ops.qa import compare_unbounded_vs_bounded

    rng = np.random.default_rng(seed)
    hi = float(rng.uniform(0.5, 40.0))
    kwargs = {**_FAST_KWARGS, "seed": seed, "y_scale": y_scale, "y_center": y_center}
    unbounded, bounded = compare_unbounded_vs_bounded(-np.inf, hi, **kwargs)
    _assert_finite_metrics(unbounded)
    _assert_finite_metrics(bounded)
    _assert_bounded_support(bounded, -np.inf, hi)
    _assert_matched_seed_reproducible(
        compare_unbounded_vs_bounded, -np.inf, hi, **kwargs
    )


@pytest.mark.parametrize(
    "seed,y_scale,y_center",
    [
        (501, 1.0, 3.0),
        (502, 15.0, -1.0),
        (503, 60.0, 10.0),
    ],
)
def test_compare_unbounded_vs_bounded_random_a_inf(seed, y_scale, y_center):
    from ops.qa import compare_unbounded_vs_bounded

    rng = np.random.default_rng(seed)
    lo = float(rng.uniform(-30.0, 5.0))
    kwargs = {**_FAST_KWARGS, "seed": seed, "y_scale": y_scale, "y_center": y_center}
    unbounded, bounded = compare_unbounded_vs_bounded(lo, np.inf, **kwargs)
    _assert_finite_metrics(unbounded)
    _assert_finite_metrics(bounded)
    _assert_bounded_support(bounded, lo, np.inf)
    _assert_matched_seed_reproducible(
        compare_unbounded_vs_bounded, lo, np.inf, **kwargs
    )
