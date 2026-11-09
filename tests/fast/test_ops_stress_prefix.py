from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner


def test_make_prefix_queries_shares_leading_features_within_runs():
    from ops.stress import make_prefix_queries

    x = make_prefix_queries(12, num_dim=6, tail=2, run_length=4, seed=1)
    assert x.shape == (12, 6)
    for start in (0, 4, 8):
        block = x[start : start + 4]
        assert np.all(block[:, :4] == block[0, :4])
        assert len({tuple(r) for r in block[:, 4:]}) == 4
    assert not np.array_equal(x[0, :4], x[4, :4])


@pytest.mark.parametrize(("tail", "run"), [(0, 4), (7, 4), (2, 0)])
def test_make_prefix_queries_rejects_bad_arguments(tail, run):
    from ops.stress import make_prefix_queries

    with pytest.raises(ValueError, match="must be"):
        make_prefix_queries(8, num_dim=6, tail=tail, run_length=run, seed=0)


def test_run_prefix_stress_reports_identical_posteriors():
    from enn._rust import prefix_reuse
    from ops.stress import format_prefix_row, run_prefix_stress

    rows = list(
        run_prefix_stress(
            ns=(60,),
            tails=(1, 3),
            num_dim=6,
            batch_rows=16,
            run_length=4,
            repeats=1,
        )
    )
    assert [r.tail for r in rows] == [1, 3]
    assert all(r.identical and r.baseline_s > 0 and r.prefix_s > 0 for r in rows)
    assert "identical = True" in format_prefix_row(rows[0])
    assert prefix_reuse() is True


def test_prefix_cli_runs_and_rejects_bad_tails():
    from ops.stress import cli

    ok = CliRunner().invoke(
        cli,
        [
            "prefix",
            "--ns",
            "40",
            "--tails",
            "2",
            "--num-dim",
            "5",
            "--batch-rows",
            "8",
            "--run-length",
            "4",
            "--repeats",
            "1",
        ],
    )
    assert ok.exit_code == 0, ok.output
    assert "D-P = 2" in ok.output
    bad = CliRunner().invoke(cli, ["prefix", "--tails", "99", "--num-dim", "5"])
    assert bad.exit_code != 0
    assert "tail must be <= num_dim" in bad.output
