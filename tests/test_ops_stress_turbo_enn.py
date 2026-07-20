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
        format_turbo_enn_row(1, 0.1234, 0.01, 0.02, n_width=2) == " 1 0.123 0.010 0.020"
    )


def test_run_turbo_enn_stress_dgp_tell_then_ask_ignore():
    events: list[str] = []
    tell_sizes: list[int] = []
    ask_returns_used: list[object] = []

    class FakeOpt:
        def ask(self, num_arms=1):
            events.append(f"ask:{num_arms}")
            arms = np.full((1, 2), 99.0, dtype=float)
            ask_returns_used.append(arms)
            return arms

        def tell(self, x, y):
            x = np.asarray(x)
            y = np.asarray(y)
            events.append(f"tell:{x.shape[0]}")
            tell_sizes.append(int(x.shape[0]))
            assert y.shape == (x.shape[0], 1)
            assert x.shape[1] == 2

    class FakeObjective:
        bounds = [-1.0, 1.0]

        def __call__(self, x):
            x = np.asarray(x)
            events.append(f"eval:{x.shape[0]}")
            return np.zeros(x.shape[0], dtype=float)

    rows = list(
        run_turbo_enn_stress(
            index_driver=ENNIndexDriver.FLAT,
            num_dim=2,
            num_obs=10,
            num_ask=3,
            optimizer=FakeOpt(),
            objective=FakeObjective(),
            bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
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

    tell_sizes: list[int] = []

    class FakeOpt:
        def ask(self, num_arms=1):
            return np.zeros((1, 2), dtype=float)

        def tell(self, x, y):
            tell_sizes.append(int(np.asarray(x).shape[0]))

    class FakeObjective:
        bounds = [-1.0, 1.0]

        def __call__(self, x):
            return np.zeros(np.asarray(x).shape[0], dtype=float)

    list(
        run_turbo_enn_stress(
            index_driver=ENNIndexDriver.FLAT,
            num_dim=2,
            num_obs=num_obs,
            num_ask=num_ask,
            optimizer=FakeOpt(),
            objective=FakeObjective(),
            bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        )
    )
    assert tell_sizes == gaps
    assert sum(tell_sizes) == num_obs


def test_turbo_enn_cli_flat_mocked(monkeypatch):
    from ops.stress import TurboEnnRoundResult, cli

    def fake_run(*, index_driver, num_dim, num_obs, num_ask, **kwargs):
        assert num_obs == 30
        assert num_ask == 4
        assert num_dim == 4
        yield TurboEnnRoundResult(n=1, iter_s=0.1, ask_s=0.04, tell_s=0.06)
        yield TurboEnnRoundResult(n=3, iter_s=0.2, ask_s=0.05, tell_s=0.15)
        yield TurboEnnRoundResult(n=10, iter_s=0.3, ask_s=0.06, tell_s=0.24)
        yield TurboEnnRoundResult(n=30, iter_s=0.4, ask_s=0.07, tell_s=0.33)

    monkeypatch.setattr("ops.stress.run_turbo_enn_stress", fake_run)
    result = CliRunner().invoke(cli, ["turbo-enn", "flat", "30", "4", "--num-dim", "4"])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=4 num_obs=30 num_ask=4 index_type=flat"
    assert lines[1:] == [
        " 1 0.100 0.040 0.060",
        " 3 0.200 0.050 0.150",
        "10 0.300 0.060 0.240",
        "30 0.400 0.070 0.330",
    ]
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
    """Large default chunk cuts per-tell overhead for N>=10000 gaps."""
    import inspect

    from ops.stress import TURBO_ENN_SEED_CHUNK

    assert TURBO_ENN_SEED_CHUNK == 20_000
    sig = inspect.signature(run_turbo_enn_stress)
    assert sig.parameters["seed_chunk"].default == TURBO_ENN_SEED_CHUNK


def test_turbo_enn_larger_seed_chunk_not_slower_on_large_gap():
    """Metamorphic: fewer tell chunks must not increase wall time for a large gap."""
    import time

    from enn import create_optimizer
    from enn.benchmarks import Ackley
    from ops.stress import (
        TURBO_ENN_ACKLEY_NOISE,
        TURBO_ENN_SEED,
        build_turbo_enn_optimizer_config,
        seed_turbo_enn_to_n,
    )

    def timed_gap(chunk: int) -> float:
        cfg = build_turbo_enn_optimizer_config(index_driver=ENNIndexDriver.FLAT)
        ackley = Ackley(
            noise=TURBO_ENN_ACKLEY_NOISE, rng=np.random.default_rng(TURBO_ENN_SEED)
        )
        bounds = np.array([ackley.bounds] * 10, dtype=float)
        opt = create_optimizer(
            bounds=bounds, config=cfg, rng=np.random.default_rng(TURBO_ENN_SEED)
        )
        rng = np.random.default_rng(TURBO_ENN_SEED + 17)
        seed_turbo_enn_to_n(opt, ackley, bounds, 5_000, rng=rng, chunk=chunk)
        t0 = time.perf_counter()
        seed_turbo_enn_to_n(opt, ackley, bounds, 20_000, rng=rng, chunk=chunk)
        return time.perf_counter() - t0

    slow = timed_gap(1_000)
    fast = timed_gap(20_000)
    assert fast <= slow * 0.75, (
        f"chunk=20k {fast:.3f}s should beat chunk=1k {slow:.3f}s by >=25%"
    )
