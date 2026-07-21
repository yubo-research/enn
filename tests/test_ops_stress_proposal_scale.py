from __future__ import annotations

import re

import numpy as np
import pytest
from click.testing import CliRunner

from enn.turbo.config.enn_index_driver import ENNIndexDriver
from ops.stress import (
    PROPOSAL_SCALE_SEED_CHUNK,
    PROPOSAL_SCALE_WARMUP,
    TURBO_ENN_NUM_INIT,
    format_proposal_scale_config_header,
    format_proposal_scale_row,
    probe_turbo_enn_proposal,
    proposal_scale_ns,
    run_proposal_scale_stress,
    seed_turbo_enn_to_n,
)
from tests.ops_stress_cli_helpers import assert_stress_cli_rejects

_PS_ROW_RE = re.compile(r" *\d+ \d+\.\d{3} \d+\.\d{3} \d+\.\d{3}")


class _TellRecorder:
    """Records tell batch shapes for seed-chunk partition tests."""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, int]] = []

    def tell(self, x, y) -> None:
        assert x.shape[0] == y.shape[0]
        assert y.shape[1] == 1
        self.shapes.append((int(x.shape[0]), int(x.shape[1])))


class _ProbeOpt:
    """Records ask/tell order for warmup and probe timing tests."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.tells = 0

    def ask(self, num_arms=1):
        self.events.append(f"ask:{num_arms}")
        return np.zeros((1, 2), dtype=float)

    def tell(self, x, y) -> None:
        self.events.append("tell")
        self.tells += int(np.asarray(x).shape[0])
        if np.asarray(x).shape[0] == 1:
            assert x.shape == (1, 2)
            assert y.shape == (1, 1)


class _ZeroObjective:
    bounds = [-1.0, 1.0]

    def __call__(self, x):
        x = np.asarray(x)
        return np.zeros(x.shape[0], dtype=float)


class _EventObjective:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self, x):
        self.events.append("eval")
        return np.array([1.0])


def test_proposal_scale_ns_filters_and_rejects():
    assert proposal_scale_ns(1000) == (10, 30, 100, 300, 1000)
    assert proposal_scale_ns(TURBO_ENN_NUM_INIT) == (10,)
    assert proposal_scale_ns(100000)[-1] == 100000
    with pytest.raises(ValueError, match="max_n must be >="):
        proposal_scale_ns(TURBO_ENN_NUM_INIT - 1)
    with pytest.raises(ValueError, match="max_n must be >="):
        proposal_scale_ns(0)


def test_seed_turbo_enn_to_n_chunk_partition():
    opt = _TellRecorder()
    bounds = np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=float)
    seed_turbo_enn_to_n(
        opt,
        _ZeroObjective(),
        bounds,
        2500,
        rng=np.random.default_rng(0),
        chunk=1000,
    )
    sizes = [s[0] for s in opt.shapes]
    assert sum(sizes) == 2500
    assert all(s <= 1000 for s in sizes)
    assert sizes == [1000, 1000, 500]
    assert all(dim == 2 for _, dim in opt.shapes)


def test_seed_turbo_enn_to_n_does_not_materialize_full_n():
    """Seed buffers must stay O(chunk), not O(n), so large-N disk stress stays RAM-bounded."""
    n = 50_000
    chunk = 1_000
    peak_obj_rows = {"n": 0}
    peak_uniform_rows = {"n": 0}

    def counting_objective(x: np.ndarray) -> np.ndarray:
        rows = int(np.asarray(x).shape[0])
        peak_obj_rows["n"] = max(peak_obj_rows["n"], rows)
        assert rows <= chunk, f"objective saw {rows} rows; chunk={chunk}"
        return np.zeros((rows, 1), dtype=float)

    inner_rng = np.random.default_rng(0)

    def bounded_uniform(*args, **kwargs):
        size = kwargs.get("size", args[2] if len(args) > 2 else None)
        if size is not None and len(size) >= 1:
            peak_uniform_rows["n"] = max(peak_uniform_rows["n"], int(size[0]))
            assert int(size[0]) <= chunk, f"uniform size {size}; chunk={chunk}"
        return inner_rng.uniform(*args, **kwargs)

    from types import SimpleNamespace

    seed_turbo_enn_to_n(
        _TellRecorder(),
        counting_objective,
        np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=float),
        n,
        rng=SimpleNamespace(uniform=bounded_uniform),
        chunk=chunk,
    )
    assert peak_obj_rows["n"] == chunk
    assert peak_uniform_rows["n"] == chunk


@pytest.mark.parametrize(
    "n, chunk, expected",
    [
        (10, 1000, [10]),
        (1000, 1000, [1000]),
        (1001, 1000, [1000, 1]),
        (3, 1, [1, 1, 1]),
    ],
)
def test_seed_turbo_enn_to_n_partition_invariants(n, chunk, expected):
    opt = _TellRecorder()
    bounds = np.array([[0.0, 1.0]] * 3, dtype=float)
    seed_turbo_enn_to_n(
        opt,
        _ZeroObjective(),
        bounds,
        n,
        rng=np.random.default_rng(1),
        chunk=chunk,
    )
    sizes = [s[0] for s in opt.shapes]
    assert sizes == expected
    assert sum(sizes) == n
    assert all(s >= 1 for s in sizes)
    assert all(s <= chunk for s in sizes)


def test_probe_turbo_enn_proposal_warmup_untimed_and_means(monkeypatch):
    events: list[str] = []
    # Timed rounds use round indices 2,3,4 → ask 0.012, 0.013, 0.014; tell 0.02.
    call_i = {"i": 0}

    def fake_perf_counter():
        idx = call_i["i"]
        call_i["i"] += 1
        phase = idx % 4
        round_idx = idx // 4
        base = round_idx * 1.0
        if phase == 0:
            return base
        if phase == 1:
            return base + 0.01 + round_idx * 0.001
        if phase == 2:
            return base + 0.5
        return base + 0.5 + 0.02

    monkeypatch.setattr("ops.stress.time.perf_counter", fake_perf_counter)
    ask_mean, tell_mean, proposal_mean = probe_turbo_enn_proposal(
        _ProbeOpt(events),
        _EventObjective(events),
        warmup=2,
        num_probes=3,
    )
    assert events == ["ask:1", "eval", "tell"] * (2 + 3)
    expected_ask = (0.012 + 0.013 + 0.014) / 3
    expected_tell = 0.02
    assert ask_mean == pytest.approx(expected_ask)
    assert tell_mean == pytest.approx(expected_tell)
    assert proposal_mean == pytest.approx(expected_ask + expected_tell)


def test_probe_turbo_enn_proposal_rejects_bad_num_probes():
    with pytest.raises(ValueError, match="num_probes must be >="):
        probe_turbo_enn_proposal(object(), lambda x: np.array([0.0]), num_probes=0)


def test_format_proposal_scale_header_and_row():
    assert (
        format_proposal_scale_config_header(
            num_dim=10, max_n=1000, num_probes=30, index_type="flat"
        )
        == "num_dim=10 max_n=1000 num_probes=30 index_type=flat"
    )
    assert (
        format_proposal_scale_config_header(
            num_dim=10,
            max_n=1000,
            num_probes=30,
            index_type="bpann_disk",
            work_dir="/tmp/w",
        )
        == "num_dim=10 max_n=1000 num_probes=30 index_type=bpann_disk work_dir=/tmp/w"
    )
    assert (
        format_proposal_scale_row(100, 0.1234, 0.01, 0.1334, n_width=4)
        == " 100 0.123 0.010 0.133"
    )


def test_run_proposal_scale_stress_uses_factory_and_grid(tmp_path):
    seen_ns: list[int] = []
    seen_work_dirs: list[str | None] = []
    work_root = tmp_path / "ps"

    def factory(*, n, work_dir):
        seen_ns.append(n)
        seen_work_dirs.append(work_dir)
        return _ProbeOpt([])

    rows = list(
        run_proposal_scale_stress(
            ENNIndexDriver.FLAT,
            2,
            30,
            work_dir=str(work_root),
            num_probes=1,
            warmup=0,
            seed_chunk=PROPOSAL_SCALE_SEED_CHUNK,
            optimizer_factory=factory,
            objective=_ZeroObjective(),
        )
    )
    assert [r.n for r in rows] == [10, 30]
    assert seen_ns == [10, 30]
    assert seen_work_dirs == [str(work_root / "n10"), str(work_root / "n30")]
    assert all(r.proposal_s == pytest.approx(r.ask_s + r.tell_s) for r in rows)
    assert (work_root / "n10").is_dir()
    assert (work_root / "n30").is_dir()


def test_proposal_scale_cli_flat_mocked(monkeypatch):
    from ops.stress import ProposalScaleResult, cli

    def fake_run(index_driver, num_dim, max_n, **kwargs):
        assert max_n == 30
        assert kwargs["num_probes"] == 2
        yield ProposalScaleResult(n=10, ask_s=0.1, tell_s=0.2, proposal_s=0.3)
        yield ProposalScaleResult(n=30, ask_s=0.11, tell_s=0.21, proposal_s=0.32)

    monkeypatch.setattr("ops.stress.run_proposal_scale_stress", fake_run)
    result = CliRunner().invoke(
        cli, ["proposal-scale", "flat", "--max-n", "30", "--num-probes", "2"]
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == "num_dim=10 max_n=30 num_probes=2 index_type=flat"
    assert lines[1:] == ["10 0.100 0.200 0.300", "30 0.110 0.210 0.320"]
    for line in lines[1:]:
        assert _PS_ROW_RE.fullmatch(line), line


@pytest.mark.parametrize(
    "args, fragment",
    [
        (["proposal-scale", "bpann_disk", "--max-n", "1000"], "requires --work-dir"),
        (
            ["proposal-scale", "flat", "--work-dir", "/tmp/x", "--max-n", "1000"],
            "work_dir requires index_type in",
        ),
        (["proposal-scale", "flat", "--max-n", "5"], "max_n must be >="),
        (["proposal-scale", "flat", "--num-dim", "0"], "num_dim must be >= 1"),
        (["proposal-scale", "flat", "--num-probes", "0"], "num_probes must be >="),
    ],
)
def test_proposal_scale_cli_rejects(args, fragment):
    assert_stress_cli_rejects(args, fragment)


def test_proposal_scale_cli_bpann_disk_mocked(tmp_path, monkeypatch):
    from ops.stress import ProposalScaleResult, cli

    work_dir = tmp_path / "ps"
    work_dir.mkdir()

    def fake_run(index_driver, num_dim, max_n, **kwargs):
        assert kwargs["work_dir"] == str(work_dir)
        yield ProposalScaleResult(n=10, ask_s=1.0, tell_s=2.0, proposal_s=3.0)

    monkeypatch.setattr("ops.stress.run_proposal_scale_stress", fake_run)
    result = CliRunner().invoke(
        cli,
        [
            "proposal-scale",
            "bpann_disk",
            "--max-n",
            "10",
            "--num-probes",
            "1",
            "--work-dir",
            str(work_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0] == (
        f"num_dim=10 max_n=10 num_probes=1 index_type=bpann_disk work_dir={work_dir}"
    )
    assert lines[1] == "10 1.000 2.000 3.000"


def test_constants_match_plan():
    assert PROPOSAL_SCALE_WARMUP == 2
    assert PROPOSAL_SCALE_SEED_CHUNK == 1000
