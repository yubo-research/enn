from __future__ import annotations

import re

import numpy as np
import pytest
from click.testing import CliRunner

from enn.turbo.config import UCBAcquisitionConfig
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from ops.stress import (
    TURBO_ENN_EVAL_SLEEP_S,
    TURBO_ENN_NUM_FIT_SAMPLES,
    TURBO_ENN_NUM_INIT,
    build_turbo_enn_optimizer_config,
    format_turbo_enn_config_header,
    format_turbo_enn_row,
    run_turbo_enn_stress,
)
from tests.ops_stress_cli_helpers import assert_stress_cli_rejects

_TURBO_ROW_RE = re.compile(r" *\d+ \d+\.\d{3} \d+\.\d{3} \d+\.\d{3}")


def test_build_turbo_enn_optimizer_config_flat():
    cfg = build_turbo_enn_optimizer_config(index_driver=ENNIndexDriver.FLAT)
    assert cfg.init.num_init == TURBO_ENN_NUM_INIT
    assert isinstance(cfg.acquisition, UCBAcquisitionConfig)
    assert cfg.surrogate.k == 10
    assert cfg.surrogate.num_fit_samples == TURBO_ENN_NUM_FIT_SAMPLES
    assert cfg.surrogate.index_driver == ENNIndexDriver.FLAT
    assert cfg.surrogate.enn_storage is None
    assert cfg.surrogate.work_dir is None
    assert cfg.trust_region.noise_aware is True


def test_build_turbo_enn_optimizer_config_bpann_disk_requires_work_dir():
    with pytest.raises(ValueError, match="bpann_disk requires work_dir"):
        build_turbo_enn_optimizer_config(index_driver=ENNIndexDriver.BPANN_DISK)


def test_build_turbo_enn_optimizer_config_bpann_disk_sets_storage():
    cfg = build_turbo_enn_optimizer_config(
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir="/tmp/turbo_enn_work",
    )
    assert cfg.surrogate.index_driver == ENNIndexDriver.BPANN_DISK
    assert cfg.surrogate.enn_storage == "disk"
    assert cfg.surrogate.work_dir == "/tmp/turbo_enn_work"


def test_build_turbo_enn_optimizer_config_rejects_work_dir_for_flat():
    with pytest.raises(ValueError, match="work_dir requires bpann_disk"):
        build_turbo_enn_optimizer_config(
            index_driver=ENNIndexDriver.FLAT,
            work_dir="/tmp/x",
        )


def test_format_turbo_enn_header_and_row():
    assert (
        format_turbo_enn_config_header(num_dim=10, num_rounds=12, index_type="flat")
        == "num_dim=10 num_rounds=12 index_type=flat"
    )
    assert (
        format_turbo_enn_config_header(
            num_dim=10,
            num_rounds=12,
            index_type="bpann_disk",
            work_dir="/tmp/w",
        )
        == "num_dim=10 num_rounds=12 index_type=bpann_disk work_dir=/tmp/w"
    )
    assert (
        format_turbo_enn_row(1, 0.1234, 0.01, 0.02, n_width=2) == " 1 0.123 0.010 0.020"
    )


def test_run_turbo_enn_stress_call_order_and_sleep():
    events: list[str] = []
    sleep_args: list[float] = []

    class FakeOpt:
        def ask(self, num_arms=1):
            events.append(f"ask:{num_arms}")
            return np.zeros((1, 2), dtype=float)

        def tell(self, x, y):
            events.append("tell")
            assert x.shape == (1, 2)
            assert y.shape == (1, 1)

    def fake_objective(x):
        events.append("eval")
        return np.array([1.0])

    def fake_sleep(seconds):
        events.append("sleep")
        sleep_args.append(seconds)

    rows = list(
        run_turbo_enn_stress(
            index_driver=ENNIndexDriver.FLAT,
            num_dim=2,
            num_rounds=TURBO_ENN_NUM_INIT,
            optimizer=FakeOpt(),
            objective=fake_objective,
            sleep_fn=fake_sleep,
        )
    )
    assert len(rows) == TURBO_ENN_NUM_INIT
    assert events == ["ask:1", "eval", "sleep", "tell"] * TURBO_ENN_NUM_INIT
    assert sleep_args == [TURBO_ENN_EVAL_SLEEP_S] * TURBO_ENN_NUM_INIT
    assert all(r.round_idx == i for i, r in enumerate(rows, start=1))


def test_run_turbo_enn_stress_rejects_short_num_rounds():
    with pytest.raises(ValueError, match="num_rounds must be >="):
        list(
            run_turbo_enn_stress(
                index_driver=ENNIndexDriver.FLAT,
                num_dim=2,
                num_rounds=TURBO_ENN_NUM_INIT - 1,
                optimizer=object(),
                objective=lambda x: np.array([0.0]),
                sleep_fn=lambda _s: None,
            )
        )


def test_turbo_enn_cli_flat(monkeypatch):
    from ops.stress import cli

    monkeypatch.setattr("ops.stress.time.sleep", lambda _s: None)
    result = CliRunner().invoke(cli, ["turbo-enn", "flat", "11", "--num-dim", "4"])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=4 num_rounds=11 index_type=flat"
    assert len(lines) == 12
    for line in lines[1:]:
        assert _TURBO_ROW_RE.fullmatch(line), line
        parts = line.split()
        assert len(parts) == 4
        float(parts[1])
        float(parts[2])
        float(parts[3])


def test_turbo_enn_cli_bpann_disk(tmp_path, monkeypatch):
    from ops.stress import cli

    monkeypatch.setattr("ops.stress.time.sleep", lambda _s: None)
    work_dir = tmp_path / "turbo_enn_cli"
    result = CliRunner().invoke(
        cli,
        [
            "turbo-enn",
            "bpann_disk",
            "11",
            "--num-dim",
            "4",
            "--work-dir",
            str(work_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == (
        f"num_dim=4 num_rounds=11 index_type=bpann_disk work_dir={work_dir}"
    )
    assert len(lines) == 12
    for line in lines[1:]:
        assert _TURBO_ROW_RE.fullmatch(line), line


@pytest.mark.parametrize(
    "args, fragment",
    [
        (["turbo-enn", "flat", "5"], "num_rounds must be >="),
        (["turbo-enn", "bpann_disk", "12"], "bpann_disk requires --work-dir"),
        (
            ["turbo-enn", "flat", "12", "--work-dir", "/tmp/x"],
            "work_dir requires index_type in",
        ),
        (["turbo-enn", "flat", "11", "--num-dim", "0"], "num_dim must be >= 1"),
    ],
)
def test_turbo_enn_cli_rejects(args, fragment):
    assert_stress_cli_rejects(args, fragment)
