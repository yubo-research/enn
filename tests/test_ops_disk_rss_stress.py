from __future__ import annotations

import pytest


def _run_disk_rss_stress(
    tmp_path, num_obs: int, *, num_dim: int = 10, query_n: int = 50
):
    from ops.stress import DiskRssStressConfig, run_disk_rss_stress

    work_dir = tmp_path / f"enn_disk_rss_{num_obs}"
    return run_disk_rss_stress(
        num_obs=num_obs,
        work_dir=str(work_dir),
        config=DiskRssStressConfig(num_dim=num_dim, query_n=query_n, batch_size=500),
    )


def _disk_rss_ceiling(num_dim: int):
    from ops.stress import DEFAULT_SHARD_MAX_ROWS, disk_stress_rss_ceiling_bytes

    return disk_stress_rss_ceiling_bytes(
        num_dim=num_dim, shard_max_rows=DEFAULT_SHARD_MAX_ROWS
    )


def _assert_disk_rss_below_ceiling(result, *, num_dim: int, num_obs: int) -> None:
    ceiling = _disk_rss_ceiling(num_dim)
    assert result.rss_delta_bytes < ceiling, (
        f"RSS delta {result.rss_delta_bytes} >= ceiling {ceiling} "
        f"(baseline={result.baseline_rss_bytes} final={result.final_rss_bytes})"
    )
    assert result.index_memory_bytes > 0
    assert result.num_obs == num_obs
    print(
        "disk_rss_stress "
        f"N={num_obs} delta={result.rss_delta_bytes} "
        f"ceiling={ceiling} index_mem={result.index_memory_bytes}"
    )


@pytest.mark.parametrize("num_obs", [1_000, 2_000])
def test_disk_rss_stress_rss_below_ceiling(tmp_path, num_obs: int):
    num_dim = 10
    result = _run_disk_rss_stress(tmp_path, num_obs, num_dim=num_dim)
    _assert_disk_rss_below_ceiling(result, num_dim=num_dim, num_obs=num_obs)


def test_disk_rss_stress_train_x_on_disk(tmp_path):
    num_dim = 10
    num_obs = 1_000
    result = _run_disk_rss_stress(tmp_path, num_obs, num_dim=num_dim)
    expected_train_x = num_obs * num_dim * 8
    assert abs(result.train_x_bytes - expected_train_x) <= num_dim * 8
    _assert_disk_rss_below_ceiling(result, num_dim=num_dim, num_obs=num_obs)


def test_disk_rss_stress_crosses_flush_threshold(tmp_path):
    """N just above default 1000-row pending threshold still stays under RSS ceiling."""
    num_dim = 10
    num_obs = 1_001
    result = _run_disk_rss_stress(tmp_path, num_obs, num_dim=num_dim)
    _assert_disk_rss_below_ceiling(result, num_dim=num_dim, num_obs=num_obs)


def test_disk_rss_stress_metamorphic_doubling_n(tmp_path):
    """RSS delta at N=2000 should stay below the same ceiling as N=1000."""
    num_dim = 10
    ceiling = _disk_rss_ceiling(num_dim)
    deltas = [
        _run_disk_rss_stress(tmp_path, num_obs, num_dim=num_dim).rss_delta_bytes
        for num_obs in (1_000, 2_000)
    ]
    assert all(delta < ceiling for delta in deltas)
    print(f"disk_rss_metamorphic deltas={deltas} ceiling={ceiling}")
