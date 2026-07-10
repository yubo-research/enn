from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.config.enn_index_driver import ENNIndexDriver

pytestmark = pytest.mark.slow


def _make_small_disk_bpann_store(work_dir, *, num_obs: int = 10, num_dim: int = 4):
    from enn.enn.enn_class import EpistemicNearestNeighbors

    from ops.stress import make_synthetic_observations

    train_x, train_y = make_synthetic_observations(num_obs, num_dim=num_dim, seed=0)
    EpistemicNearestNeighbors(
        train_x,
        train_y,
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )


def test_make_uniform_query_points_shape_and_bounds():
    from ops.stress import make_uniform_query_points

    x = make_uniform_query_points(50, num_dim=3, seed=7)
    assert x.shape == (50, 3)
    assert np.all(x >= -1.0)
    assert np.all(x <= 1.0)
    x2 = make_uniform_query_points(50, num_dim=3, seed=7)
    np.testing.assert_allclose(x, x2)


def test_run_sample_stress_on_disk_store(tmp_path):
    from ops.stress import SampleStressConfig, run_sample_stress

    work_dir = tmp_path / "enn_sample"
    _make_small_disk_bpann_store(work_dir, num_obs=10, num_dim=4)
    result = run_sample_stress(
        work_dir=str(work_dir),
        config=SampleStressConfig(num_samples=5, seed=1),
    )
    assert result.num_dim == 4
    assert result.num_obs == 10
    assert result.num_samples == 5
    assert result.draws_shape == (1, 5, 1)
    assert result.all_finite
    assert result.init_s >= 0.0
    assert result.sample_s >= 0.0


def test_sample_stress_cli_on_disk_store(tmp_path):
    from click.testing import CliRunner

    from ops.stress import cli

    work_dir = tmp_path / "enn_sample_cli"
    _make_small_disk_bpann_store(work_dir, num_obs=10, num_dim=4)
    result = CliRunner().invoke(
        cli,
        ["sample", str(work_dir), "5", "--seed", "1"],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == (
        f"num_dim=4 num_obs=10 work_dir={work_dir} num_samples=5 seed=1"
    )
    assert lines[1].startswith(
        "draws_shape=(1, 5, 1) function_seeds=1 all_finite=true init_s="
    )
    assert " sample_s=" in lines[1]


def test_sample_stress_cli_rejects_missing_work_dir():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        ["sample", "/nonexistent/enn_store", "5"],
    )
    assert result.exit_code != 0
    assert "work_dir does not exist" in result.output


def test_sample_stress_cli_rejects_missing_required_args():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["sample"])
    assert result.exit_code != 0


def test_sample_stress_cli_rejects_invalid_num_samples(tmp_path):
    from click.testing import CliRunner

    from ops.stress import cli

    work_dir = tmp_path / "enn_sample_invalid"
    _make_small_disk_bpann_store(work_dir)
    result = CliRunner().invoke(
        cli,
        ["sample", str(work_dir), "0"],
    )
    assert result.exit_code != 0
    assert "num_samples must be >= 1" in result.output
