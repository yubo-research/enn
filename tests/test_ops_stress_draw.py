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
            num_samples=4,
        )
    )
    assert np.isfinite(result.posterior.avg_likelihood)
    assert np.isfinite(result.posterior_function_draw.avg_likelihood)
    assert np.isfinite(result.posterior.argmin_rms)
    assert np.isfinite(result.posterior_function_draw.argmin_rms)
    assert result.posterior.argmin_rms >= 0.0
    assert result.posterior_function_draw.argmin_rms >= 0.0
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
    assert result.num_samples == 4


def test_draw_stress_cli_happy_path():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        [
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
            "--num-samples",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("num_dim=2 num_obs=40 num_test=20 seed=0 k=5")
    assert "num_samples=4" in lines[0]
    assert "epistemic_variance_scale=" in lines[0]
    assert "aleatoric_variance_scale=" in lines[0]
    assert lines[1].startswith("posterior avg_likelihood=")
    assert lines[2].startswith("posterior_function_draw avg_likelihood=")
    assert "argmin_rms=" in lines[1]
    assert "argmin_rms=" in lines[2]
    assert "draws_shape=" in lines[1]
    assert "draws_shape=" in lines[2]
    avg_post = float(lines[1].split("avg_likelihood=")[1].split()[0])
    avg_fn = float(lines[2].split("avg_likelihood=")[1].split()[0])
    rms_post = float(lines[1].split("argmin_rms=")[1].split()[0])
    rms_fn = float(lines[2].split("argmin_rms=")[1].split()[0])
    assert np.isfinite(avg_post)
    assert np.isfinite(avg_fn)
    assert np.isfinite(rms_post)
    assert np.isfinite(rms_fn)
    assert rms_post >= 0.0
    assert rms_fn >= 0.0


def test_draw_stress_cli_default_num_samples_is_100():
    from click.testing import CliRunner

    from ops.stress import DEFAULT_DRAW_NUM_SAMPLES, cli

    assert DEFAULT_DRAW_NUM_SAMPLES == 100
    result = CliRunner().invoke(cli, ["draw", "--help"])
    assert result.exit_code == 0, result.output
    assert "100" in result.output
    assert "--num-samples" in result.output


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
