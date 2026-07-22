from __future__ import annotations

import re

import numpy as np
import pytest
from click.testing import CliRunner

from enn.turbo.config import UCBAcquisitionConfig
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from ops.stress import (
    TURBO_ENN_NUM_FIT_SAMPLES,
    TURBO_ENN_NUM_INIT,
    build_turbo_enn_optimizer_config,
    format_turbo_enn_config_header,
    format_turbo_enn_row,
    run_turbo_enn_stress,
    turbo_enn_ask_stops,
)
from tests.ops_stress_cli_helpers import assert_stress_cli_rejects
from tests.ops_stress_turbo_enn_helpers import (
    RecordingObjective,
    RecordingTurboOpt,
    collect_tell_sizes,
    unit_bounds,
)

_TURBO_ROW_RE = re.compile(r" *\d+ \d+\.\d{3} \d+\.\d{3} \d+\.\d{3}")


def test_turbo_enn_ask_stops_plan_example():
    assert turbo_enn_ask_stops(30, 4) == (1, 3, 10, 30)


def test_turbo_enn_ask_stops_invariants():
    for num_obs, num_ask in ((1, 1), (10, 1), (10, 10), (100, 5), (30, 4), (7, 3)):
        stops = turbo_enn_ask_stops(num_obs, num_ask)
        assert len(stops) == num_ask
        assert stops[-1] == num_obs
        assert stops[0] >= 1
        assert all(stops[i] < stops[i + 1] for i in range(len(stops) - 1))
        assert all(1 <= s <= num_obs for s in stops)


@pytest.mark.parametrize(
    "num_obs, num_ask, fragment",
    [
        (0, 1, "num_obs must be >="),
        (10, 0, "num_ask must be >="),
        (5, 6, "num_ask must be <="),
    ],
)
def test_turbo_enn_ask_stops_rejects(num_obs, num_ask, fragment):
    with pytest.raises(ValueError, match=fragment):
        turbo_enn_ask_stops(num_obs, num_ask)


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
        format_turbo_enn_config_header(
            num_dim=10, num_obs=30, num_ask=4, index_type="flat"
        )
        == "num_dim=10 num_obs=30 num_ask=4 index_type=flat"
    )
    assert (
        format_turbo_enn_config_header(
            num_dim=10,
            num_obs=30,
            num_ask=4,
            index_type="bpann_disk",
            work_dir="/tmp/w",
        )
        == "num_dim=10 num_obs=30 num_ask=4 index_type=bpann_disk work_dir=/tmp/w"
    )
    assert (
        format_turbo_enn_config_header(
            num_dim=2,
            num_obs=10,
            num_ask=3,
            index_type="flat",
            tell_all=True,
        )
        == "num_dim=2 num_obs=10 num_ask=3 index_type=flat tell_all=true"
    )
    assert format_turbo_enn_config_header(
        num_dim=2,
        num_obs=10,
        num_ask=3,
        index_type="bpann_disk",
        work_dir="/tmp/w",
        tell_all=True,
    ) == (
        "num_dim=2 num_obs=10 num_ask=3 index_type=bpann_disk "
        "work_dir=/tmp/w tell_all=true"
    )
    assert "tell_all" not in format_turbo_enn_config_header(
        num_dim=2, num_obs=10, num_ask=3, index_type="flat", tell_all=False
    )
    assert (
        format_turbo_enn_row(1, 0.1234, 0.01, 0.02, n_width=2) == " 1 0.123 0.010 0.020"
    )


def test_run_turbo_enn_stress_dgp_tell_then_ask_ignore():
    events: list[str] = []
    tell_sizes: list[int] = []
    ask_returns_used: list[object] = []
    opt = RecordingTurboOpt(
        tell_sizes=tell_sizes,
        events=events,
        ask_returns_used=ask_returns_used,
    )
    rows = list(
        run_turbo_enn_stress(
            index_driver=ENNIndexDriver.FLAT,
            num_dim=2,
            num_obs=10,
            num_ask=3,
            optimizer=opt,
            objective=RecordingObjective(events=events),
            bounds=unit_bounds(2),
            seed_chunk=100,
        )
    )
    stops = turbo_enn_ask_stops(10, 3)
    assert [r.n for r in rows] == list(stops)
    gaps = [stops[0]] + [stops[i] - stops[i - 1] for i in range(1, len(stops))]
    assert tell_sizes == gaps
    expected_events: list[str] = []
    for gap in gaps:
        expected_events.extend([f"eval:{gap}", f"tell:{gap}", "ask:1"])
    assert events == expected_events
    # Ask arms must not be fed back into tell (tell sizes == DGP gaps only).
    assert sum(tell_sizes) == 10
    assert all(r.iter_s == pytest.approx(r.ask_s + r.tell_s) for r in rows)
    assert len(ask_returns_used) == len(stops)


def test_run_turbo_enn_stress_rejects_bad_num_ask():
    with pytest.raises(ValueError, match="num_ask must be <="):
        list(
            run_turbo_enn_stress(
                index_driver=ENNIndexDriver.FLAT,
                num_dim=2,
                num_obs=3,
                num_ask=4,
                optimizer=object(),
                objective=lambda x: np.zeros(np.asarray(x).shape[0]),
            )
        )


@pytest.mark.parametrize(
    "num_obs, num_ask", [(30, 4), (1, 1), (50, 7), (8, 8), (100, 3)]
)
def test_turbo_enn_gaps_sum_to_num_obs(num_obs, num_ask):
    stops = turbo_enn_ask_stops(num_obs, num_ask)
    gaps = [stops[0]] + [stops[i] - stops[i - 1] for i in range(1, len(stops))]
    assert sum(gaps) == num_obs
    assert all(g >= 1 for g in gaps)

    tell_sizes = collect_tell_sizes(num_obs=num_obs, num_ask=num_ask, tell_all=False)
    assert tell_sizes == gaps
    assert sum(tell_sizes) == num_obs


def test_run_turbo_enn_stress_tell_all_one_row_per_tell():
    """tell_all=True → every tell has one row; tell count equals num_obs."""
    events: list[str] = []
    tell_sizes: list[int] = []
    opt = RecordingTurboOpt(tell_sizes=tell_sizes, events=events)
    num_obs, num_ask = 10, 3
    rows = list(
        run_turbo_enn_stress(
            index_driver=ENNIndexDriver.FLAT,
            num_dim=2,
            num_obs=num_obs,
            num_ask=num_ask,
            optimizer=opt,
            objective=RecordingObjective(events=events),
            bounds=unit_bounds(2),
            seed_chunk=100,  # must lose to tell_all
            tell_all=True,
        )
    )
    stops = turbo_enn_ask_stops(num_obs, num_ask)
    gaps = [stops[0]] + [stops[i] - stops[i - 1] for i in range(1, len(stops))]
    assert [r.n for r in rows] == list(stops)
    assert tell_sizes == [1] * num_obs
    assert sum(tell_sizes) == num_obs
    expected_events: list[str] = []
    for gap in gaps:
        expected_events.extend(["eval:1", "tell:1"] * gap)
        expected_events.append("ask:1")
    assert events == expected_events
    assert all(r.iter_s == pytest.approx(r.ask_s + r.tell_s) for r in rows)


def test_run_turbo_enn_stress_tell_all_overrides_invalid_seed_chunk():
    """tell_all forces seed_chunk=1 before validation, so seed_chunk=0 is fine."""
    tell_sizes = collect_tell_sizes(num_obs=5, num_ask=2, tell_all=True, seed_chunk=0)
    assert tell_sizes == [1] * 5


def test_run_turbo_enn_stress_unit_gaps_match_with_or_without_tell_all():
    """When every gap is already 1, tell streams match; only header advertises flag."""
    num_obs = num_ask = 4
    without = collect_tell_sizes(num_obs=num_obs, num_ask=num_ask, tell_all=False)
    with_flag = collect_tell_sizes(num_obs=num_obs, num_ask=num_ask, tell_all=True)
    assert without == with_flag == [1] * num_obs
    assert "tell_all" not in format_turbo_enn_config_header(
        num_dim=2, num_obs=num_obs, num_ask=num_ask, index_type="flat"
    )
    assert (
        format_turbo_enn_config_header(
            num_dim=2,
            num_obs=num_obs,
            num_ask=num_ask,
            index_type="flat",
            tell_all=True,
        )
        == f"num_dim=2 num_obs={num_obs} num_ask={num_ask} index_type=flat tell_all=true"
    )


def test_turbo_enn_cli_flat_mocked(monkeypatch):
    from ops.stress import TurboEnnRoundResult, cli

    seen: dict[str, object] = {}

    def fake_run(*, index_driver, num_dim, num_obs, num_ask, **kwargs):
        seen.update(kwargs)
        assert num_obs == 30
        assert num_ask == 4
        assert num_dim == 4
        assert kwargs.get("tell_all") is False
        assert "seed_chunk" not in kwargs
        yield TurboEnnRoundResult(n=1, iter_s=0.1, ask_s=0.04, tell_s=0.06)
        yield TurboEnnRoundResult(n=3, iter_s=0.2, ask_s=0.05, tell_s=0.15)
        yield TurboEnnRoundResult(n=10, iter_s=0.3, ask_s=0.06, tell_s=0.24)
        yield TurboEnnRoundResult(n=30, iter_s=0.4, ask_s=0.07, tell_s=0.33)

    monkeypatch.setattr("ops.stress.run_turbo_enn_stress", fake_run)
    result = CliRunner().invoke(cli, ["turbo-enn", "flat", "30", "4", "--num-dim", "4"])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=4 num_obs=30 num_ask=4 index_type=flat"
    assert "tell_all" not in lines[0]
    assert lines[1:] == [
        " 1 0.100 0.040 0.060",
        " 3 0.200 0.050 0.150",
        "10 0.300 0.060 0.240",
        "30 0.400 0.070 0.330",
    ]
    for line in lines[1:]:
        assert _TURBO_ROW_RE.fullmatch(line), line


def test_turbo_enn_cli_tell_all_mocked(monkeypatch):
    from ops.stress import TurboEnnRoundResult, cli

    received: dict[str, object] = {}

    def fake_run(*, index_driver, num_dim, num_obs, num_ask, **kwargs):
        received["tell_all"] = kwargs.get("tell_all")
        received["kwargs"] = kwargs
        assert num_obs == 10
        assert num_ask == 3
        assert num_dim == 2
        assert "seed_chunk" not in kwargs
        yield TurboEnnRoundResult(n=1, iter_s=0.1, ask_s=0.04, tell_s=0.06)
        yield TurboEnnRoundResult(n=3, iter_s=0.2, ask_s=0.05, tell_s=0.15)
        yield TurboEnnRoundResult(n=10, iter_s=0.3, ask_s=0.06, tell_s=0.24)

    monkeypatch.setattr("ops.stress.run_turbo_enn_stress", fake_run)
    result = CliRunner().invoke(
        cli, ["turbo-enn", "flat", "10", "3", "--num-dim", "2", "--tell-all"]
    )
    assert result.exit_code == 0, result.output
    assert received["tell_all"] is True
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=2 num_obs=10 num_ask=3 index_type=flat tell_all=true"
    assert len(lines) == 4
    for line in lines[1:]:
        assert _TURBO_ROW_RE.fullmatch(line), line


def test_turbo_enn_cli_bpann_disk_mocked(tmp_path, monkeypatch):
    from ops.stress import TurboEnnRoundResult, cli

    work_dir = tmp_path / "turbo_enn_cli"
    work_dir.mkdir()

    def fake_run(*, index_driver, num_dim, num_obs, num_ask, **kwargs):
        assert kwargs["work_dir"] == str(work_dir)
        assert num_obs == 10
        assert num_ask == 2
        yield TurboEnnRoundResult(n=3, iter_s=1.0, ask_s=0.4, tell_s=0.6)
        yield TurboEnnRoundResult(n=10, iter_s=2.0, ask_s=0.5, tell_s=1.5)

    monkeypatch.setattr("ops.stress.run_turbo_enn_stress", fake_run)
    result = CliRunner().invoke(
        cli,
        [
            "turbo-enn",
            "bpann_disk",
            "10",
            "2",
            "--num-dim",
            "4",
            "--work-dir",
            str(work_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == (
        f"num_dim=4 num_obs=10 num_ask=2 index_type=bpann_disk work_dir={work_dir}"
    )
    assert lines[1:] == [" 3 1.000 0.400 0.600", "10 2.000 0.500 1.500"]


@pytest.mark.parametrize(
    "args, fragment",
    [
        (["turbo-enn", "flat", "5", "6"], "num_ask must be <="),
        (["turbo-enn", "flat", "0", "1"], "num_obs must be >="),
        (["turbo-enn", "flat", "10", "0"], "num_ask must be >="),
        (["turbo-enn", "bpann_disk", "12", "3"], "bpann_disk requires --work-dir"),
        (
            ["turbo-enn", "flat", "12", "3", "--work-dir", "/tmp/x"],
            "work_dir requires index_type in",
        ),
        (["turbo-enn", "flat", "11", "2", "--num-dim", "0"], "num_dim must be >= 1"),
    ],
)
def test_turbo_enn_cli_rejects(args, fragment):
    assert_stress_cli_rejects(args, fragment)


def test_turbo_enn_cli_real_optimizer_smoke():
    from ops.stress import cli

    result = CliRunner().invoke(cli, ["turbo-enn", "flat", "10", "3", "--num-dim", "2"])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=2 num_obs=10 num_ask=3 index_type=flat"
    stops = turbo_enn_ask_stops(10, 3)
    assert len(lines) == 1 + len(stops)
    for line, stop in zip(lines[1:], stops, strict=True):
        assert _TURBO_ROW_RE.fullmatch(line), line
        assert int(line.split()[0]) == stop


def test_turbo_enn_seed_chunk_default_is_large():
    """Flat uses a larger default chunk than disk; both beat tiny proposal-scale chunks."""
    import inspect

    from ops.stress import (
        TURBO_ENN_SEED_CHUNK,
        TURBO_ENN_SEED_CHUNK_FLAT,
        turbo_enn_default_seed_chunk,
    )

    assert TURBO_ENN_SEED_CHUNK == 100_000
    assert TURBO_ENN_SEED_CHUNK_FLAT == 200_000
    assert TURBO_ENN_SEED_CHUNK_FLAT > TURBO_ENN_SEED_CHUNK
    assert (
        turbo_enn_default_seed_chunk(ENNIndexDriver.FLAT) == TURBO_ENN_SEED_CHUNK_FLAT
    )
    assert (
        turbo_enn_default_seed_chunk(ENNIndexDriver.BPANN_DISK) == TURBO_ENN_SEED_CHUNK
    )
    sig = inspect.signature(run_turbo_enn_stress)
    assert sig.parameters["seed_chunk"].default is None


def _turbo_enn_seed_gap_tell_count(gap: int, chunk: int) -> int:
    """Count ``tell`` calls while bulk-seeding ``gap`` points with ``chunk``."""
    from unittest.mock import MagicMock

    from enn import create_optimizer
    from enn.benchmarks import Ackley
    from ops.stress import (
        TURBO_ENN_ACKLEY_NOISE,
        TURBO_ENN_SEED,
        build_turbo_enn_optimizer_config,
        seed_turbo_enn_to_n,
    )

    cfg = build_turbo_enn_optimizer_config(index_driver=ENNIndexDriver.FLAT)
    ackley = Ackley(
        noise=TURBO_ENN_ACKLEY_NOISE, rng=np.random.default_rng(TURBO_ENN_SEED)
    )
    bounds = np.array([ackley.bounds] * 10, dtype=float)
    opt = create_optimizer(
        bounds=bounds, config=cfg, rng=np.random.default_rng(TURBO_ENN_SEED)
    )
    counter = MagicMock(side_effect=opt.tell)
    opt.tell = counter  # type: ignore[method-assign]
    rng = np.random.default_rng(TURBO_ENN_SEED + 17)
    seed_turbo_enn_to_n(opt, ackley, bounds, gap, rng=rng, chunk=chunk)
    return int(counter.call_count)


def test_turbo_enn_larger_seed_chunk_fewer_tells_on_large_gap():
    """Metamorphic: larger seed_chunk ⇒ fewer tell() calls on a fixed gap (no wall clock)."""
    from ops.stress import TURBO_ENN_SEED_CHUNK_FLAT

    gap = 1_000
    small_chunk = 100
    assert TURBO_ENN_SEED_CHUNK_FLAT > gap, "flat default chunk must cover the test gap"

    small = _turbo_enn_seed_gap_tell_count(gap, small_chunk)
    large = _turbo_enn_seed_gap_tell_count(gap, TURBO_ENN_SEED_CHUNK_FLAT)
    assert small == (gap + small_chunk - 1) // small_chunk
    assert large == 1
    assert large < small, (
        f"chunk={TURBO_ENN_SEED_CHUNK_FLAT} → {large} tells; "
        f"chunk={small_chunk} → {small} tells; expected fewer tells"
    )


def test_turbo_enn_flat_chunk_covers_medium_gap_in_one_tell():
    """Flat default chunk covers a mid-size gap in one tell; disk still chunks it."""
    from ops.stress import TURBO_ENN_SEED_CHUNK, TURBO_ENN_SEED_CHUNK_FLAT

    gap = 150_000
    assert TURBO_ENN_SEED_CHUNK < gap < TURBO_ENN_SEED_CHUNK_FLAT
    flat_tells = _turbo_enn_seed_gap_tell_count(gap, TURBO_ENN_SEED_CHUNK_FLAT)
    disk_tells = _turbo_enn_seed_gap_tell_count(gap, TURBO_ENN_SEED_CHUNK)
    assert flat_tells == 1
    assert disk_tells == (gap + TURBO_ENN_SEED_CHUNK - 1) // TURBO_ENN_SEED_CHUNK
    assert flat_tells < disk_tells
