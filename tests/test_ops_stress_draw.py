from __future__ import annotations

import numpy as np
import pytest


def test_draw_f_1d_and_multid():
    from ops.stress import draw_f

    x1 = np.array([[0.3], [0.5]])
    y1 = draw_f(x1)
    assert y1.shape == (2, 1)
    np.testing.assert_allclose(y1, [[0.0], [0.04]])

    x2 = np.array([[0.3, 0.3], [0.4, 0.2]])
    y2 = draw_f(x2)
    assert y2.shape == (2, 1)
    np.testing.assert_allclose(y2, [[0.0], [0.01 + 0.01]])


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
        )
    )
    assert np.isfinite(result.avg_likelihood)
    assert result.epistemic_variance_scale > 0.0
    assert result.aleatoric_variance_scale >= 0.0
    assert result.num_obs == 40
    assert result.num_test == 20
    assert result.num_dim == 2


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
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=2 num_obs=40 num_test=20 seed=0 k=5"
    assert "avg_likelihood=" in lines[1]
    assert "epistemic_variance_scale=" in lines[1]
    assert "aleatoric_variance_scale=" in lines[1]
    avg = float(lines[1].split("avg_likelihood=")[1].split()[0])
    assert np.isfinite(avg)


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
