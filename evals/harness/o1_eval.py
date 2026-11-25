from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from enn import _rust
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fitter import ENNStatefulFitter
from evals.harness import o1_measure as ms
from evals.harness.metrics import format_larger, format_smaller, gaussian_loglik, rmse
from evals.harness.o1_workloads import CHUNK_ROWS, QUERY_MODES, Workload, queries, rows
from evals.harness.scaling_n import dir_mib, memory_mib, reset_peak
from evals.harness.work_dirs import scoped_work_dir

STORES = ("O1", "O1F", "OLOGN")


@dataclass(frozen=True)
class O1EvalConfig:
    workload: Workload = field(default_factory=Workload)
    store: str = "O1"
    n_grid: tuple[int, ...] = (1000, 3000, 10000)
    num_queries: int = 100
    k: int = 10
    insert_batch: int = 1
    predict: bool = True
    num_fit_candidates: int = 16
    num_fit_samples: int = 32
    work_parent: str | None = None


def config_fields(config: O1EvalConfig) -> dict:
    out = asdict(config)
    out.pop("work_parent")
    out["workload"] = config.workload.label()
    out.update({f"wl_{k}": v for k, v in asdict(config.workload).items()})
    return out


FOREST_FIELDS = (
    "mean_depth",
    "disk_nodes",
    "free_nodes",
    "node_bytes",
    "splits",
    "rebuilds",
    "rebuilt_rows",
    "max_rebuild_rows",
    "global_rebuilds",
)


def node_io_per_row(before, after, num_rows: int) -> dict[str, float]:
    if before is None or after is None or num_rows <= 0:
        return {}
    return {
        "insert_node_reads_per_row": (after[0] - before[0]) / num_rows,
        "insert_node_writes_per_row": (after[1] - before[1]) / num_rows,
    }


def timed_search(backend, q: np.ndarray, k: int):
    lat, got_d, got_i = [], [], []
    for i in range(len(q)):
        t0 = time.perf_counter()
        d2, idx = backend.neighbor_distances_and_indices(q[i : i + 1], k)
        lat.append(time.perf_counter() - t0)
        got_d.append(d2[0])
        got_i.append(idx[0])
    return lat, np.asarray(got_d), np.asarray(got_i)


def candidate_stats(cands, exact_idx: np.ndarray) -> dict[str, float]:
    if cands is None:
        return {}
    return {
        "candidate_recall": ms.candidate_recall(exact_idx, cands),
        "candidates_mean": float(np.mean([len(c) for c in cands])),
    }


class O1EvalRun:
    def __init__(self, work_dir: str, config: O1EvalConfig) -> None:
        self.work_dir = work_dir
        self.config = config
        self.model: EpistemicNearestNeighbors | None = None
        self.fitter = ENNStatefulFitter(k=config.k, seed=config.workload.seed + 1)
        self.base = memory_mib()
        self.num_rows = 0

    def _build(self, hi: int) -> None:
        first = min(hi, CHUNK_ROWS)
        xs, ys = zip(*[(x, y) for _, x, y in rows(self.config.workload, 0, first)])
        self.model = EpistemicNearestNeighbors(
            np.vstack(xs),
            np.vstack(ys),
            work_dir=self.work_dir,
            weighting=True,
            store=self.config.store,
        )
        self.model.ensure_index_sync()
        self.num_rows = first

    def insert(self, hi: int) -> dict[str, float]:
        if self.model is None:
            self._build(hi)
        lo, batch = self.num_rows, self.config.insert_batch
        lat, io0, t_all = [], ms.io_counters(), time.perf_counter()
        nodes0 = _rust.index_node_io(self.model.rust_backend)
        for _, x, y in rows(self.config.workload, lo, hi):
            for b in range(0, len(x), batch):
                t0 = time.perf_counter()
                self.model.add(x[b : b + batch], y[b : b + batch])
                self.model.ensure_index_sync()
                lat.append(time.perf_counter() - t0)
        elapsed = time.perf_counter() - t_all
        self.num_rows = hi
        out = {
            "inserted": hi - lo,
            "insert_rows_per_s": (hi - lo) / max(elapsed, 1e-12),
        }
        out.update(ms.latency_stats("insert", lat))
        out.update(ms.io_delta(io0, ms.io_counters(), hi - lo, "insert_io_per_row"))
        out.update(
            node_io_per_row(
                nodes0, _rust.index_node_io(self.model.rust_backend), hi - lo
            )
        )
        return out

    def weights(self) -> np.ndarray:
        return 1.0 / np.asarray(self.model._x_scale, dtype=float).reshape(-1) ** 2

    def query_mode(self, mode: str) -> dict[str, float]:
        cfg = self.config
        q, _ = queries(cfg.workload, self.num_rows, cfg.num_queries, mode, 0)
        backend = self.model.rust_backend
        io0, nodes0 = ms.io_counters(), _rust.index_node_io(backend)
        lat, got_d, got_i = timed_search(backend, q, cfg.k)
        io1, nodes1 = ms.io_counters(), _rust.index_node_io(backend)
        t0 = time.perf_counter()
        ex_d, ex_i = ms.exact_knn(cfg.workload, self.num_rows, q, self.weights(), cfg.k)
        out = {
            "exact_query_us": 1e6 * (time.perf_counter() - t0) / len(q),
            "recall": ms.recall(ex_i, got_i),
            "metric_ok": ms.metric_consistent(ex_d, got_d),
            "qps": len(q) / max(sum(lat), 1e-12),
        }
        out.update(ms.latency_stats("query", lat))
        out.update(ms.io_delta(io0, io1, len(q), "query_io"))
        if nodes0 is not None:
            out["query_node_reads"] = (nodes1[0] - nodes0[0]) / len(q)
        out.update(candidate_stats(_rust.index_candidates(backend, q, cfg.k), ex_i))
        return {f"{mode}_{k}": v for k, v in out.items()}

    def invariants(self) -> dict[str, float]:
        out = {"stored_frac": len(self.model) / self.num_rows}
        occ = _rust.index_occupancy(self.model.rust_backend)
        if occ is None:
            return out
        used = sum(occ["keys_used"])
        out.update(
            reachable_frac=occ["reachable_rows"] / self.num_rows,
            hash_bits=occ["hash_bits"],
            bucket_cap=occ["bucket_cap"],
            max_arrivals=occ["max_arrivals"],
            overflow_key_frac=sum(occ["overflow_keys"]) / max(used, 1),
            table_rows_frac=float(np.mean(occ["table_rows"])) / self.num_rows,
        )
        if occ["node_bytes"]:
            out.update({key: occ[key] for key in FOREST_FIELDS})
        return out

    def memory(self) -> dict[str, float]:
        now = memory_mib()
        disk = dir_mib(self.work_dir)
        return {
            "rss_mib": now["rss_mib"] - self.base["rss_mib"],
            "peak_mib": now["peak_mib"] - self.base["rss_mib"],
            "index_mem_mib": self.model.index_memory_bytes() / 2**20,
            "disk_mib": disk,
            "disk_bytes_per_row": disk * 2**20 / self.num_rows,
            "work_fs": ms.fs_type(self.work_dir),
        }

    def prediction(self) -> dict[str, float]:
        cfg = self.config
        x, y = queries(cfg.workload, self.num_rows, cfg.num_queries, "data", 1)
        t0 = time.perf_counter()
        params = self.fitter.ask(
            self.model,
            num_fit_candidates=cfg.num_fit_candidates,
            num_fit_samples=cfg.num_fit_samples,
            affine_calibrate=True,
        )
        fit_s = time.perf_counter() - t0
        post = self.model.posterior(x, params=params)
        return {
            "fit_s": fit_s,
            "nrmse": rmse(y, post.mu) / max(float(np.std(y)), 1e-12),
            "loglik": gaussian_loglik(y, post.mu, post.se),
        }

    def checkpoint(self, hi: int) -> dict:
        reset_peak()
        row = {**config_fields(self.config), "n": hi, **self.insert(hi)}
        for mode in QUERY_MODES:
            row.update(self.query_mode(mode))
        row.update(self.invariants())
        if self.config.predict:
            row.update(self.prediction())
        row.update(self.memory())
        return row


def summary_line(row: dict) -> str:
    parts = [
        f"workload={row['workload']} store={row['store']} n={row['n']}",
        format_larger("recall", f"{row['data_recall']:.3f}"),
        format_larger("incumbent_recall", f"{row['incumbent_recall']:.3f}"),
        format_smaller("query_p99_us", f"{row['data_query_p99_us']:.0f}"),
        format_smaller("insert_p99_us", f"{row.get('insert_p99_us', 0.0):.0f}"),
    ]
    if "reachable_frac" in row:
        parts.append(format_larger("reachable", f"{row['reachable_frac']:.3f}"))
    if "nrmse" in row:
        parts.append(format_smaller("nrmse", f"{row['nrmse']:.3f}"))
    return " ".join(parts)


def run_config(config: O1EvalConfig, out_path: str | None = None) -> list[dict]:
    results = []
    with scoped_work_dir(
        "enn_o1_eval_", ms.work_parent(config.work_parent)
    ) as work_dir:
        run = O1EvalRun(work_dir, config)
        for hi in config.n_grid:
            results.append(run.checkpoint(hi))
            emit(results[-1], out_path)
    return results


def emit(row: dict, out_path: str | None) -> None:
    print(summary_line(row), flush=True)
    write_row(row, out_path)


def write_row(row: dict, out_path: str | None) -> None:
    line = json.dumps(row)
    if out_path is None:
        print(line, flush=True)
        return
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
