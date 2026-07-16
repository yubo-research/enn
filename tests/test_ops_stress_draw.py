from __future__ import annotations

import numpy as np
import pytest


def test_draw_f_1d_and_multid():
    from ops.stress import DRAW_F_CENTER, draw_f

    x1 = np.array([[DRAW_F_CENTER], [0.5]])
    y1 = draw_f(x1)
    assert y1.shape == (2, 1)
    np.testing.assert_allclose(y1, [[0.0], [0.04]])

    x2 = np.array([[DRAW_F_CENTER, DRAW_F_CENTER], [0.4, 0.2]])
    y2 = draw_f(x2)
    assert y2.shape == (2, 1)
    np.testing.assert_allclose(y2, [[0.0], [0.01 + 0.01]])


def test_argmin_rms_known_minima():
    from ops.stress import DRAW_F_CENTER, argmin_rms

    # Two test points in 2D; three samples pick indices 0, 1, 0.
    x_test = np.array([[0.3, 0.3], [0.5, 0.1]], dtype=float)
    # draws (B=2, M=1, S=3): sample0 min at i=0, sample1 at i=1, sample2 at i=0
    draws = np.array(
        [
            [[0.0, 2.0, 0.1]],
            [[1.0, 0.0, 0.2]],
        ],
        dtype=float,
    )
    # residuals: [0,0], [0.2,-0.2], [0,0] vs center 0.3
    # ||eps||^2: 0, 0.08, 0 -> mean 0.08/3 -> rms sqrt(0.08/3)
    expected = float(np.sqrt(0.08 / 3.0))
    assert argmin_rms(x_test, draws) == pytest.approx(expected)
    # sanity: center constant matches DRAW_F_CENTER
    assert DRAW_F_CENTER == pytest.approx(0.3)


def test_argmin_hit_rate_known_minima():
    from ops.stress import argmin_hit_rate

    # True f-argmin is index 0 (at center); draws pick 0, 1, 0 -> hit rate 2/3.
    x_test = np.array([[0.3, 0.3], [0.5, 0.1]], dtype=float)
    draws = np.array(
        [
            [[0.0, 2.0, 0.1]],
            [[1.0, 0.0, 0.2]],
        ],
        dtype=float,
    )
    assert argmin_hit_rate(x_test, draws) == pytest.approx(2.0 / 3.0)


def test_make_draw_observations_bounds_and_reproducible():
    from ops.stress import make_draw_observations

    rng = np.random.default_rng(7)
    x, y = make_draw_observations(20, num_dim=3, rng=rng)
    assert x.shape == (20, 3)
    assert y.shape == (20, 1)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)

    x_a, y_a = make_draw_observations(15, num_dim=2, rng=np.random.default_rng(3))
    x_b, y_b = make_draw_observations(15, num_dim=2, rng=np.random.default_rng(3))
    np.testing.assert_allclose(x_a, x_b)
    np.testing.assert_allclose(y_a, y_b)


def test_gaussian_and_average_likelihood():
    from ops.stress import average_likelihood, gaussian_likelihood

    y = np.array([[0.0], [1.0]])
    mu = np.array([[0.0], [1.0]])
    se = np.array([[1.0], [1.0]])
    lik = gaussian_likelihood(y, mu, se)
    expected = 1.0 / np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(lik, [[expected], [expected]])
    assert average_likelihood(y, mu, se) == pytest.approx(expected)


def test_average_likelihood_from_draws():
    from ops.stress import average_likelihood_from_draws

    y = np.array([[0.0], [1.0]])
    # (batch, metrics, num_samples)
    draws = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0]],
        ],
        dtype=float,
    )
    # empirical se is 0 -> floored; density is large but finite
    avg = average_likelihood_from_draws(y, draws)
    assert np.isfinite(avg)
    assert avg > 0.0


def test_run_draw_stress_finite_likelihood():
    from ops.stress import DrawStressConfig, run_draw_stress

    result = run_draw_stress(
        DrawStressConfig(
            num_obs=40,
            num_test=20,
            num_dim=2,
            seed=0,
            k=5,
            num_fit_candidates=8,
            num_fit_samples=5,
            num_draws=4,
        )
    )
    assert np.isfinite(result.posterior.avg_likelihood)
    assert np.isfinite(result.posterior_function_draw.avg_likelihood)
    assert np.isfinite(result.posterior.argmin_rms)
    assert np.isfinite(result.posterior_function_draw.argmin_rms)
    assert result.posterior.argmin_rms >= 0.0
    assert result.posterior_function_draw.argmin_rms >= 0.0
    assert 0.0 <= result.posterior.argmin_hit_rate <= 1.0
    assert 0.0 <= result.posterior_function_draw.argmin_hit_rate <= 1.0
    assert result.posterior.method == "posterior"
    assert result.posterior_function_draw.method == "posterior_function_draw"
    assert result.posterior.all_finite
    assert result.posterior_function_draw.all_finite
    assert result.posterior.draws_shape == (20, 1, 4)
    assert result.posterior_function_draw.draws_shape == (20, 1, 4)
    assert result.epistemic_variance_scale > 0.0
    assert result.aleatoric_variance_scale >= 0.0
    assert result.num_obs == 40
    assert result.num_test == 20
    assert result.num_dim == 2
    assert result.num_draws == 4


def test_mean_se_and_format():
    from ops.stress import MeanSE, format_mean_se, mean_se

    one = mean_se([2.0])
    assert one.mean == pytest.approx(2.0)
    assert not np.isfinite(one.se)
    assert format_mean_se(one) == "2"

    two = mean_se([1.0, 3.0])
    assert two.mean == pytest.approx(2.0)
    assert two.se == pytest.approx(
        1.0
    )  # std=sqrt(2)/1? ddof=1: std=sqrt(2), se=sqrt(2)/sqrt(2)=1
    assert format_mean_se(two) == "2 ± 1"
    assert format_mean_se(MeanSE(0.2193, 0.0123), fmt="0.4f") == "0.2193 ± 0.0123"


def test_run_draw_stress_over_seeds_aggregates():
    from ops.stress import DrawStressConfig, run_draw_stress_over_seeds

    agg = run_draw_stress_over_seeds(
        DrawStressConfig(
            num_obs=40,
            num_test=20,
            num_dim=2,
            seed=0,
            k=5,
            num_fit_candidates=8,
            num_fit_samples=5,
            num_draws=4,
        ),
        num_seeds=3,
    )
    assert agg.num_seeds == 3
    assert agg.seed == 0
    assert np.isfinite(agg.posterior.avg_likelihood.mean)
    assert np.isfinite(agg.posterior.avg_likelihood.se)
    assert np.isfinite(agg.posterior_function_draw.argmin_hit_rate.mean)
    assert np.isfinite(agg.posterior_function_draw.argmin_hit_rate.se)
    assert agg.posterior.argmin_hit_rate.se >= 0.0


def _tiny_draw_cli_args(*, num_seeds: int | None = None) -> list[str]:
    args = [
        "draw",
        "40",
        "20",
        "--num-dim",
        "2",
        "--seed",
        "0",
        "--k",
        "5",
        "--num-fit-candidates",
        "8",
        "--num-fit-samples",
        "5",
        "--num-draws",
        "4",
    ]
    if num_seeds is not None:
        args.extend(["--num-seeds", str(num_seeds)])
    return args


def _assert_draw_cli_metric_line(line: str, method: str, *, with_se: bool) -> None:
    assert line.startswith(f"{method} avg_likelihood=")
    assert "argmin_rms=" in line
    assert "argmin_hit_rate=" in line
    if with_se:
        assert " ± " in line
    else:
        assert " ± " not in line
        assert "draws_shape=" not in line
        assert "all_finite=" not in line
    avg = float(line.split("avg_likelihood=")[1].split()[0])
    rms = float(line.split("argmin_rms=")[1].split()[0])
    hit_tok = line.split("argmin_hit_rate=")[1].split()[0]
    if with_se:
        hit_tok = line.split("argmin_hit_rate=")[1].split("eval_s=")[0].strip()
        mean_s, se_s = hit_tok.split(" ± ")
        hit = float(mean_s)
        assert mean_s == f"{hit:0.4f}"
        assert se_s == f"{float(se_s):0.4f}"
    else:
        hit = float(hit_tok)
        assert hit_tok == f"{hit:0.4f}"
    assert np.isfinite(avg)
    assert np.isfinite(rms)
    assert rms >= 0.0
    assert 0.0 <= hit <= 1.0


def test_draw_stress_cli_happy_path():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, _tiny_draw_cli_args())
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("num_dim=2 num_obs=40 num_test=20 seed=0")
    assert "num_seeds=1" in lines[0]
    assert "num_draws=4" in lines[0]
    assert "epistemic_variance_scale=" in lines[0]
    assert "aleatoric_variance_scale=" in lines[0]
    _assert_draw_cli_metric_line(lines[1], "posterior", with_se=False)
    _assert_draw_cli_metric_line(lines[2], "posterior_function_draw", with_se=False)


def test_draw_stress_cli_multi_seed_reports_mean_se():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, _tiny_draw_cli_args(num_seeds=3))
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 3
    assert "num_seeds=3" in lines[0]
    assert " ± " in lines[0]
    _assert_draw_cli_metric_line(lines[1], "posterior", with_se=True)
    _assert_draw_cli_metric_line(lines[2], "posterior_function_draw", with_se=True)


def test_draw_stress_cli_default_num_draws_is_100():
    from click.testing import CliRunner

    from ops.stress import DEFAULT_DRAW_NUM_DRAWS, DEFAULT_DRAW_NUM_SEEDS, cli

    assert DEFAULT_DRAW_NUM_DRAWS == 100
    assert DEFAULT_DRAW_NUM_SEEDS == 1
    result = CliRunner().invoke(cli, ["draw", "--help"])
    assert result.exit_code == 0, result.output
    assert "100" in result.output
    assert "--num-draws" in result.output
    assert "--num-seeds" in result.output
    assert "--num-samples" not in result.output


def test_draw_stress_cli_rejects_num_obs_lt_one():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["draw", "0", "10"])
    assert result.exit_code != 0
    assert "num_obs must be >= 1" in result.output


def test_draw_stress_cli_rejects_missing_args():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["draw"])
    assert result.exit_code != 0
