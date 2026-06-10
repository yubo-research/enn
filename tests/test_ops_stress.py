from __future__ import annotations

import re
import time

import numpy as np
import pytest

from enn.turbo.config.enn_index_driver import ENNIndexDriver


def _collect_synthetic_stream(
    *,
    num_obs: int,
    num_dim: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    from ops.stress import iter_synthetic_observations

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x_row, y_row in iter_synthetic_observations(
        num_obs,
        num_dim=num_dim,
        seed=seed,
        batch_size=batch_size,
    ):
        xs.append(x_row)
        ys.append(y_row)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def test_checkpoint_ns_report_grid():
    from ops.stress import checkpoint_ns

    assert checkpoint_ns(100) == (1, 3, 10, 30, 100)
    assert checkpoint_ns(100_000)[-1] == 100_000
    assert checkpoint_ns(100_000)[:5] == (1, 3, 10, 30, 100)


def test_checkpoint_ns_monotone_and_within_cap():
    from ops.stress import checkpoint_ns

    for max_n in (1, 3, 10, 30, 100, 1000, 10_000):
        ns = checkpoint_ns(max_n)
        assert ns[0] == 1
        assert ns[-1] <= max_n
        assert ns == tuple(sorted(ns))
        assert len(ns) == len(set(ns))


def test_checkpoint_ns_metamorphic_doubling_preserves_prefix():
    from ops.stress import checkpoint_ns

    for max_n in (30, 100, 1000):
        small = checkpoint_ns(max_n)
        large = checkpoint_ns(max_n * 2)
        assert small == large[: len(small)]


@pytest.mark.parametrize("name", ["flat", "hnsw", "hnsw_disk"])
def test_parse_index_driver(name: str):
    from ops.stress import parse_index_driver

    driver = parse_index_driver(name)
    assert isinstance(driver, ENNIndexDriver)


def test_make_query_points_shape_and_reproducible():
    from ops.stress import make_query_points

    q1 = make_query_points(1000, num_dim=4, seed=7)
    q2 = make_query_points(1000, num_dim=4, seed=7)
    assert q1.shape == (1000, 4)
    np.testing.assert_allclose(q1, q2)


def test_run_enn_add_stress_returns_checkpoint_query_times():
    from ops.stress import EnnAddStressConfig, checkpoint_ns, run_enn_add_stress

    num_obs = 30
    rows = list(
        run_enn_add_stress(
            index_driver=ENNIndexDriver.FLAT,
            num_obs=num_obs,
            config=EnnAddStressConfig(num_dim=4, seed=0, query_n=10, query_seed=1),
        )
    )
    assert [n for n, _, _ in rows] == list(checkpoint_ns(num_obs))
    for n, query_s, segment_s in rows:
        assert n >= 1
        assert query_s >= 0.0
        assert segment_s >= 0.0


def test_run_enn_add_stress_syncs_at_checkpoints(monkeypatch):
    from ops.stress import EnnAddStressConfig, checkpoint_ns, run_enn_add_stress

    sync_calls: list[int] = []

    def _count_sync(self):
        sync_calls.append(1)
        return None

    monkeypatch.setattr(
        "ops.stress.EpistemicNearestNeighbors.ensure_index_sync",
        _count_sync,
    )

    num_obs = 30
    list(
        run_enn_add_stress(
            index_driver=ENNIndexDriver.FLAT,
            num_obs=num_obs,
            config=EnnAddStressConfig(num_dim=4, seed=0, query_n=5, query_seed=1),
        )
    )
    assert sync_calls == [1] * len(checkpoint_ns(num_obs))


def test_run_enn_add_stress_hnsw_disk_no_checkpoint_sync(monkeypatch, tmp_path):
    from ops.stress import EnnAddStressConfig, checkpoint_ns, run_enn_add_stress

    sync_calls: list[int] = []

    def _count_sync(self):
        sync_calls.append(1)
        return None

    monkeypatch.setattr(
        "ops.stress.EpistemicNearestNeighbors.ensure_index_sync",
        _count_sync,
    )

    num_obs = 30
    list(
        run_enn_add_stress(
            index_driver=ENNIndexDriver.HNSW_DISK,
            num_obs=num_obs,
            config=EnnAddStressConfig(
                num_dim=4,
                seed=0,
                query_n=5,
                query_seed=1,
                work_dir=str(tmp_path),
            ),
        )
    )
    assert sync_calls == []
    assert len(list(checkpoint_ns(num_obs))) > 0


def test_run_enn_add_stress_hnsw_disk_schedules_flush_after_add(monkeypatch, tmp_path):
    from ops.stress import EnnAddStressConfig, run_enn_add_stress

    flush_calls: list[int] = []

    def _count_flush(self):
        flush_calls.append(1)
        return None

    monkeypatch.setattr(
        "ops.stress.EpistemicNearestNeighbors.schedule_background_flush",
        _count_flush,
    )

    num_obs = 30
    list(
        run_enn_add_stress(
            index_driver=ENNIndexDriver.HNSW_DISK,
            num_obs=num_obs,
            config=EnnAddStressConfig(
                num_dim=4,
                seed=0,
                query_n=5,
                query_seed=1,
                work_dir=str(tmp_path),
            ),
        )
    )
    assert flush_calls == [1] * num_obs


def test_run_enn_add_stress_does_not_fit(monkeypatch):
    from enn.enn.enn_fitter import ENNStatefulFitter

    tell_calls: list[int] = []
    ask_calls: list[int] = []

    def _count_tell(self, *args, **kwargs):
        tell_calls.append(1)

    def _count_ask(self, *args, **kwargs):
        ask_calls.append(1)

    monkeypatch.setattr(ENNStatefulFitter, "tell", _count_tell)
    monkeypatch.setattr(ENNStatefulFitter, "ask", _count_ask)

    from ops.stress import EnnAddStressConfig, run_enn_add_stress

    list(
        run_enn_add_stress(
            index_driver=ENNIndexDriver.FLAT,
            num_obs=30,
            config=EnnAddStressConfig(num_dim=4, seed=0, query_n=5, query_seed=1),
        )
    )
    assert tell_calls == []
    assert ask_calls == []


def test_format_config_header():
    from ops.stress import format_config_header

    assert format_config_header(num_dim=10, num_obs=100) == "num_dim=10 num_obs=100"
    assert (
        format_config_header(num_dim=10, num_obs=100, work_dir="/tmp/enn_work")
        == "num_dim=10 num_obs=100 work_dir=/tmp/enn_work"
    )


def test_stress_row_n_width():
    from ops.stress import stress_row_n_width

    assert stress_row_n_width(10) == 2
    assert stress_row_n_width(100_000) == 6


def test_format_stress_row_fixed_width_n():
    from ops.stress import format_stress_row

    row = format_stress_row(10, 1.2345, 0.0567, n_width=6)
    assert row == "    10 1.234 0.057"
    assert re.fullmatch(r" {4}10 1\.234 0\.057", row)

    row_large = format_stress_row(100_000, 0.5, 12.3, n_width=6)
    assert row_large == "100000 0.500 12.300"


def test_run_enn_add_stress_segment_excludes_query(monkeypatch):
    from ops.stress import EnnAddStressConfig, run_enn_add_stress

    query_delay_s = 0.05

    def _slow_query(model, x_query):
        time.sleep(query_delay_s)
        return query_delay_s

    monkeypatch.setattr("ops.stress._time_query_s", _slow_query)

    rows = list(
        run_enn_add_stress(
            index_driver=ENNIndexDriver.FLAT,
            num_obs=3,
            config=EnnAddStressConfig(num_dim=2, seed=0, query_n=2, query_seed=1),
        )
    )
    assert len(rows) == 2
    for _n, query_s, segment_s in rows:
        assert query_s >= query_delay_s - 1e-6
        assert segment_s < query_delay_s


_STRESS_ROW_RE = re.compile(r" *\d+ \d+\.\d{3} \d+\.\d{3}")


def test_enn_stress_cli_does_not_fit(monkeypatch):
    from click.testing import CliRunner

    from enn.enn.enn_fitter import ENNStatefulFitter
    from ops.stress import cli

    tell_calls: list[int] = []
    ask_calls: list[int] = []

    def _count_tell(self, *args, **kwargs):
        tell_calls.append(1)

    def _count_ask(self, *args, **kwargs):
        ask_calls.append(1)

    monkeypatch.setattr(ENNStatefulFitter, "tell", _count_tell)
    monkeypatch.setattr(ENNStatefulFitter, "ask", _count_ask)

    num_obs = 10
    result = CliRunner().invoke(cli, ["enn", "flat", str(num_obs)])
    assert result.exit_code == 0
    assert tell_calls == []
    assert ask_calls == []


def test_enn_stress_cli():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        ["enn", "flat", "10"],
    )
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=10 num_obs=10"
    assert len(lines) == 4
    data_lines = lines[1:]
    for line in data_lines:
        assert _STRESS_ROW_RE.fullmatch(line)
    assert data_lines[0].startswith(" ")
    assert not data_lines[-1].startswith(" ")


def test_enn_stress_cli_rejects_work_dir_for_in_memory_driver():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        ["enn", "flat", "10", "--work-dir", "/tmp/enn_work"],
    )
    assert result.exit_code != 0
    assert "work_dir requires index_type in" in result.output


def test_enn_stress_cli_rejects_disk_driver_without_work_dir():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["enn", "hnsw_disk", "10"])
    assert result.exit_code != 0
    assert "hnsw_disk requires --work-dir" in result.output


def test_enn_stress_cli_hnsw_disk(tmp_path):
    from click.testing import CliRunner

    from ops.stress import cli

    work_dir = tmp_path / "enn_cli_disk"
    result = CliRunner().invoke(
        cli,
        ["enn", "hnsw_disk", "10", "--work-dir", str(work_dir)],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == f"num_dim=10 num_obs=10 work_dir={work_dir}"
    assert len(lines) == 4
    for line in lines[1:]:
        assert _STRESS_ROW_RE.fullmatch(line)


def test_enn_stress_cli_rejects_legacy_option_syntax():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        ["enn", "--index-type", "flat", "--num-obs", "10"],
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_enn_stress_cli_rejects_swapped_positional_order():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["enn", "10", "flat"])
    assert result.exit_code != 0


def test_enn_stress_cli_num_dim_option():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(
        cli,
        ["enn", "flat", "10", "--num-dim", "4"],
    )
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=4 num_obs=10"
    assert len(lines) == 4
    for line in lines[1:]:
        assert _STRESS_ROW_RE.fullmatch(line)


def test_enn_stress_cli_rejects_invalid_num_dim():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["enn", "flat", "10", "--num-dim", "0"])
    assert result.exit_code != 0
    assert "num_dim must be >= 1" in result.output


def test_make_synthetic_observations():
    from ops.stress import make_synthetic_observations

    x, y = make_synthetic_observations(5, num_dim=3, seed=0)
    assert x.shape == (5, 3)
    assert y.shape == (5, 1)


def test_iter_synthetic_observations_matches_make_synthetic():
    num_obs, num_dim, seed = 250, 7, 0

    x_one, y_one = _collect_synthetic_stream(
        num_obs=num_obs,
        num_dim=num_dim,
        seed=seed,
        batch_size=1,
    )
    x_hundred, y_hundred = _collect_synthetic_stream(
        num_obs=num_obs,
        num_dim=num_dim,
        seed=seed,
        batch_size=100,
    )
    np.testing.assert_allclose(x_one, x_hundred)
    np.testing.assert_allclose(y_one, y_hundred)
    assert x_one.shape == (num_obs, num_dim)
    assert y_one.shape == (num_obs, 1)


def test_iter_synthetic_observation_batches_matches_make_synthetic():
    from ops.stress import (
        iter_synthetic_observation_batches,
        make_synthetic_observations,
    )

    num_obs, num_dim, seed = 250, 7, 0
    x_ref, y_ref = make_synthetic_observations(num_obs, num_dim=num_dim, seed=seed)
    batches = list(
        iter_synthetic_observation_batches(
            num_obs,
            num_dim=num_dim,
            seed=seed,
            batch_size=num_obs,
        )
    )
    assert len(batches) == 1
    x_batch, y_batch = batches[0]
    assert x_batch.shape == (num_obs, num_dim)
    assert y_batch.shape == (num_obs, 1)
    np.testing.assert_allclose(x_batch, x_ref)
    np.testing.assert_allclose(y_batch, y_ref)


def test_iter_synthetic_observations_metamorphic_batch_size():
    num_obs, num_dim, seed = 37, 4, 3

    reference_x, reference_y = _collect_synthetic_stream(
        num_obs=num_obs,
        num_dim=num_dim,
        seed=seed,
        batch_size=17,
    )
    for batch_size in (1, 7, 100):
        x_batch, y_batch = _collect_synthetic_stream(
            num_obs=num_obs,
            num_dim=num_dim,
            seed=seed,
            batch_size=batch_size,
        )
        np.testing.assert_allclose(x_batch, reference_x)
        np.testing.assert_allclose(y_batch, reference_y)


@pytest.mark.parametrize("seed", [0, 1, 11])
def test_iter_synthetic_observations_fuzz(seed: int):
    rng = np.random.default_rng(seed)
    num_obs = int(rng.integers(1, 500))
    num_dim = int(rng.integers(1, 20))
    batch_a = int(rng.integers(1, 101))
    batch_b = int(rng.integers(1, 101))
    data_seed = int(rng.integers(0, 10_000))

    x_a, y_a = _collect_synthetic_stream(
        num_obs=num_obs,
        num_dim=num_dim,
        seed=data_seed,
        batch_size=batch_a,
    )
    x_b, y_b = _collect_synthetic_stream(
        num_obs=num_obs,
        num_dim=num_dim,
        seed=data_seed,
        batch_size=batch_b,
    )
    np.testing.assert_allclose(x_a, x_b)
    np.testing.assert_allclose(y_a, y_b)
    print(
        "iter_synthetic_observations_fuzz "
        f"seed={seed} num_obs={num_obs} num_dim={num_dim} "
        f"batch_a={batch_a} batch_b={batch_b}"
    )


def test_stress_main(monkeypatch):
    from ops.stress import main

    monkeypatch.setattr(
        "sys.argv",
        ["stress", "enn", "flat", "3"],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_checkpoint_ns_fuzz(seed: int):
    from ops.stress import checkpoint_ns

    rng = __import__("numpy").random.default_rng(seed)
    max_n = int(rng.integers(1, 50_000))
    ns = checkpoint_ns(max_n)
    assert ns[0] == 1
    assert all(1 <= n <= max_n for n in ns)
    assert ns == tuple(sorted(set(ns)))
    print(f"checkpoint_ns_fuzz seed={seed} max_n={max_n} len={len(ns)}")


def test_max_rss_bytes_returns_positive():
    from ops.stress import max_rss_bytes

    assert max_rss_bytes() > 0


def test_disk_stress_rss_ceiling_grows_with_dim_not_n():
    from ops.stress import DEFAULT_SHARD_MAX_ROWS, disk_stress_rss_ceiling_bytes

    low_dim = disk_stress_rss_ceiling_bytes(
        num_dim=4, shard_max_rows=DEFAULT_SHARD_MAX_ROWS
    )
    high_dim = disk_stress_rss_ceiling_bytes(
        num_dim=40, shard_max_rows=DEFAULT_SHARD_MAX_ROWS
    )
    assert high_dim > low_dim
