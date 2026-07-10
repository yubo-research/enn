from __future__ import annotations

import json
import re
import struct

import pytest

_STRESS_ROW_RE = re.compile(r" *\d+ \d+\.\d{3} \d+(?:\.\d+)?(?:e[+-]?\d+)?")


def test_format_config_header_restart():
    from ops.stress import format_config_header

    assert (
        format_config_header(
            num_dim=10,
            num_obs=100_000_000,
            work_dir="_bpann",
            num_obs_existing=3_000_000,
        )
        == "restarting num_dim=10 num_obs=100000000 num_obs_existing=3000000 work_dir=_bpann"
    )


def test_load_num_obs_existing(tmp_path):
    from ops.stress import load_num_obs_existing

    assert load_num_obs_existing(str(tmp_path)) == 0

    meta = {
        "format_version": 1,
        "num_obs": 42,
        "num_dim": 10,
        "num_metrics": 1,
        "scale_x": False,
        "index_backend": "bpann_disk",
        "indexed_rows": 42,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    assert load_num_obs_existing(str(tmp_path)) == 42

    (tmp_path / "num_obs.bin").write_bytes(struct.pack("<Q", 7))
    assert load_num_obs_existing(str(tmp_path)) == 7


@pytest.mark.parametrize(
    "index_type,subdir",
    [("bpann_disk", "enn_restart_bpann")],
)
def test_enn_stress_cli_disk_restart_header(tmp_path, index_type, subdir):
    from click.testing import CliRunner

    from ops.stress import cli

    work_dir = tmp_path / subdir
    first = CliRunner().invoke(
        cli,
        ["enn", index_type, "3", "--work-dir", str(work_dir)],
    )
    assert first.exit_code == 0, first.output

    second = CliRunner().invoke(
        cli,
        ["enn", index_type, "10", "--work-dir", str(work_dir)],
    )
    assert second.exit_code == 0, second.output
    lines = second.output.strip().splitlines()
    assert (
        lines[0]
        == f"restarting num_dim=10 num_obs=10 num_obs_existing=3 work_dir={work_dir}"
    )
    for line in lines[1:]:
        assert _STRESS_ROW_RE.fullmatch(line)
