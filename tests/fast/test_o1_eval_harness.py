import numpy as np
import pytest

from evals.harness import o1_drift, o1_eval, o1_measure, o1_suites, o1_workloads
from evals.harness.o1_workloads import CHUNK_ROWS, Workload


@pytest.mark.parametrize("kind", o1_workloads.KINDS)
def test_rows_are_reproducible_and_chunk_aligned(kind):
    wl = Workload(kind, 6, seed=3)
    blocks = list(o1_workloads.rows(wl, 500, CHUNK_ROWS + 700))
    assert [b[0] for b in blocks] == [500, CHUNK_ROWS]
    assert sum(len(b[1]) for b in blocks) == CHUNK_ROWS + 200
    x, y = o1_workloads.chunk(wl, 0)
    np.testing.assert_array_equal(blocks[0][1], x[500:])
    assert y.shape == (CHUNK_ROWS, 1)
    assert wl.label().endswith("_d6")


def test_unknown_kind_and_mode_are_rejected():
    with pytest.raises(ValueError, match="unknown workload kind"):
        o1_workloads.geometry(Workload("missing", 3))
    with pytest.raises(ValueError, match="unknown query mode"):
        o1_workloads.queries(Workload(), 10, 2, "missing", 0)


def test_trajectory_converges_and_subspace_is_low_dimensional():
    wl = Workload("trajectory", 5)
    geo = o1_workloads.geometry(wl)
    x, _ = o1_workloads.chunk(wl, 0)
    dist = np.linalg.norm(x - geo.x_star, axis=1)
    assert dist[-1] < dist[0] / 50
    near, _ = o1_workloads.queries(wl, CHUNK_ROWS, 20, "incumbent", 0)
    assert np.linalg.norm(near - geo.x_star, axis=1).max() < 2 * dist[-1]
    sub, _ = o1_workloads.chunk(Workload("subspace", 20, intrinsic=2), 0)
    sv = np.linalg.svd(sub - sub.mean(0), compute_uv=False)
    assert sv[2] < 0.05 * sv[1]


def test_exact_knn_matches_brute_force_under_weights(monkeypatch):
    monkeypatch.setattr(o1_measure, "EXACT_BLOCK_BYTES", 1000)
    wl = Workload("mixed", 4)
    n, k = 1500, 5
    x = np.vstack([b[1] for b in o1_workloads.rows(wl, 0, n)])
    q, _ = o1_workloads.queries(wl, n, 7, "data", 0)
    w = np.array([1.0, 4.0, 0.5, 2.0])
    d2, idx = o1_measure.exact_knn(wl, n, q, w, k)
    brute = ((x[None] - q[:, None]) ** 2 * w).sum(-1)
    want = np.argsort(brute, axis=1)[:, :k]
    np.testing.assert_array_equal(idx, want)
    np.testing.assert_allclose(d2, np.take_along_axis(brute, want, 1))
    assert o1_measure.metric_consistent(d2, d2)
    assert not o1_measure.metric_consistent(d2, d2 * 0.5)


def test_recall_latency_io_and_fs_helpers(tmp_path):
    true = np.array([[0, 1], [2, 3]])
    assert o1_measure.recall(true, np.array([[1, 9], [2, 3]])) == 0.75
    assert o1_measure.candidate_recall(true, [[0, 1, 5], [3]]) == 0.75
    stats = o1_measure.latency_stats("q", [1e-6, 2e-6, 3e-6])
    assert stats["q_max_us"] == pytest.approx(3.0)
    assert stats["q_p50_us"] == pytest.approx(2.0)
    assert o1_measure.latency_stats("q", []) == {}
    io = o1_measure.io_counters()
    assert set(io) == set(o1_measure.IO_KEYS)
    assert o1_measure.io_delta(io, io, 2, "x")["x_syscr"] == 0
    assert o1_measure.work_parent(str(tmp_path / "w")).endswith("w")
    assert o1_measure.fs_type(str(tmp_path)) != ""


@pytest.mark.parametrize("store", o1_eval.STORES)
def test_run_config_reports_quality_cost_and_invariants(store, tmp_path, capsys):
    config = o1_eval.O1EvalConfig(
        workload=Workload("trajectory", 4),
        store=store,
        n_grid=(300, 600),
        num_queries=5,
        insert_batch=50,
        predict=store == "O1",
        num_fit_candidates=2,
        num_fit_samples=4,
        work_parent=str(tmp_path),
    )
    out = tmp_path / "rows.jsonl"
    rows = o1_eval.run_config(config, str(out))
    assert [r["n"] for r in rows] == [300, 600]
    last = rows[-1]
    assert last["inserted"] == 300 and last["stored_frac"] == 1.0
    assert last["data_metric_ok"] and 0.0 <= last["data_recall"] <= 1.0
    assert "insert_p99_us" in last and "data_query_p99_us" in last
    assert ("reachable_frac" in last) == (store != "OLOGN")
    assert ("mean_depth" in last) == (store == "O1F")
    if store == "O1F":
        assert last["reachable_frac"] == 1.0 and last["max_arrivals"] <= 32
        assert (
            last["insert_node_writes_per_row"] > 0
            and last["data_query_node_reads"] >= 0
        )
    assert ("nrmse" in last) == (store == "O1")
    assert len(out.read_text().splitlines()) == 2
    assert "LARGER(recall)" in capsys.readouterr().out


def test_drift_schedules_and_tiny_run(tmp_path):
    for schedule in o1_drift.SCHEDULES:
        first = o1_drift.metric_scale(schedule, 0, 4, 6)
        last = o1_drift.metric_scale(schedule, 3, 4, 6)
        assert first.shape == (6,) and np.all(last > 0)
        assert not np.allclose(first, last) or schedule == "alternating"
    with pytest.raises(ValueError, match="unknown drift schedule"):
        o1_drift.metric_scale("missing", 0, 1, 2)
    config = o1_drift.DriftConfig(
        schedule="relevant_dims",
        workload=Workload("uniform", 4),
        n_grid=(300, 600),
        num_queries=5,
        work_parent=str(tmp_path),
    )
    rows = o1_drift.run_drift(config)
    assert [r["step"] for r in rows] == [0, 1]
    assert rows[-1]["ref_metric_ok"] and rows[-1]["inc_metric_ok"]
    assert rows[-1]["scale_ratio"] == pytest.approx(10.0)


def test_suite_grids_follow_notes():
    assert o1_suites.n_grid_for(1000, o1_suites.LONG_N)[-1] == 10**6
    assert o1_suites.n_grid_for(100, o1_suites.LONG_N)[-1] == 10**7
    labels = {w.label() for w in o1_suites.long_workloads()}
    assert {"subspace3_d10", "subspace3_d100", "subspace3_d1000"} <= labels
    assert {w.kind for w in o1_suites.short_workloads()} == set(o1_workloads.KINDS)
