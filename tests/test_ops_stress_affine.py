"""CLI and draw-path coverage for stress.py --affine."""

from __future__ import annotations

from click.testing import CliRunner

from ops.stress import cli


STRESS_COMMANDS = ("enn", "sample", "draw", "turbo-enn", "proposal-scale")


def test_each_stress_command_help_lists_affine() -> None:
    for name in STRESS_COMMANDS:
        result = CliRunner().invoke(cli, [name, "--help"])
        assert result.exit_code == 0, result.output
        assert "--affine" in result.output
        assert "--no-affine" in result.output


def test_draw_cli_affine_header_and_runs() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "draw",
            "20",
            "10",
            "--num-dim",
            "3",
            "--k",
            "3",
            "--num-fit-candidates",
            "4",
            "--num-fit-samples",
            "8",
            "--num-draws",
            "8",
            "--affine",
        ],
    )
    assert result.exit_code == 0, result.output
    header = result.output.strip().splitlines()[0]
    assert "affine=true" in header


def test_enn_cli_affine_in_header(tmp_path) -> None:
    result = CliRunner().invoke(
        cli,
        ["enn", "flat", "3", "--num-dim", "2", "--heartbeat-seconds", "0", "--affine"],
    )
    assert result.exit_code == 0, result.output
    assert "affine = true" in result.output.strip().splitlines()[0]


def test_build_turbo_enn_config_forwards_affine() -> None:
    from ops.stress import build_turbo_enn_optimizer_config
    from enn.turbo.config.enn_index_driver import ENNIndexDriver

    cfg = build_turbo_enn_optimizer_config(
        index_driver=ENNIndexDriver.FLAT,
        affine_calibrate=True,
    )
    assert cfg.surrogate.fit.affine_calibrate is True


def test_mk_enn_affine_calibrate_runs() -> None:
    import numpy as np

    from enn.turbo.config import ENNFitConfig
    from enn.turbo.proposal import mk_enn

    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 2))
    y = (x.sum(axis=1) + 0.5).reshape(-1, 1)
    model, params = mk_enn(
        x,
        y,
        k=3,
        rng=rng,
        fit=ENNFitConfig(
            num_fit_samples=6, num_fit_candidates=4, affine_calibrate=True
        ),
    )
    assert model is not None and params is not None
    assert params.k_num_neighbors == 3
