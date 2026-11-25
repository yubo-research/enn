from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from evals.harness.o1_workloads import Workload, rows

EXACT_BLOCK_BYTES = 8 << 20
IO_KEYS = ("read_bytes", "write_bytes", "syscr", "syscw")
PERCENTILES = (("p50", 50.0), ("p95", 95.0), ("p99", 99.0))
WORK_PARENT_ENV = "ENN_EVAL_WORK_PARENT"
DEFAULT_WORK_PARENT = Path.home() / ".cache" / "enn_evals"


def _merge_best(best_d, best_i, d2, ids, k):
    all_d = np.concatenate([best_d, d2], axis=1)
    all_i = np.concatenate([best_i, np.broadcast_to(ids, d2.shape)], axis=1)
    keep = np.argpartition(all_d, k - 1, axis=1)[:, :k] if all_d.shape[1] > k else None
    if keep is None:
        return all_d, all_i
    return np.take_along_axis(all_d, keep, 1), np.take_along_axis(all_i, keep, 1)


def _block_dist2(x: np.ndarray, q: np.ndarray, weights: np.ndarray) -> np.ndarray:
    per_query = max(1, EXACT_BLOCK_BYTES // max(1, x.size * 8))
    out = np.empty((len(q), len(x)))
    for lo in range(0, len(q), per_query):
        diff = x[None, :, :] - q[lo : lo + per_query, None, :]
        out[lo : lo + per_query] = np.einsum("qnd,qnd,d->qn", diff, diff, weights)
    return out


def exact_knn(workload: Workload, num_rows: int, q: np.ndarray, weights, k: int):
    weights = np.asarray(weights, dtype=float).reshape(-1)
    best_d = np.empty((len(q), 0))
    best_i = np.empty((len(q), 0), dtype=np.int64)
    for start, x, _ in rows(workload, 0, num_rows):
        ids = np.arange(start, start + len(x), dtype=np.int64)[None, :]
        best_d, best_i = _merge_best(
            best_d, best_i, _block_dist2(x, q, weights), ids, k
        )
    order = np.argsort(best_d, axis=1, kind="stable")
    return np.take_along_axis(best_d, order, 1), np.take_along_axis(best_i, order, 1)


def recall(true_idx: np.ndarray, got_idx: np.ndarray) -> float:
    hits = sum(
        len(set(t.tolist()) & set(g.tolist())) for t, g in zip(true_idx, got_idx)
    )
    return hits / max(1, true_idx.size)


def candidate_recall(true_idx: np.ndarray, candidates) -> float:
    hits = sum(
        len(set(t.tolist()) & set(np.asarray(c).tolist()))
        for t, c in zip(true_idx, candidates)
    )
    return hits / max(1, true_idx.size)


def metric_consistent(exact_d2: np.ndarray, got_d2: np.ndarray) -> bool:
    got = np.sort(np.asarray(got_d2, dtype=float), axis=1)[:, : exact_d2.shape[1]]
    tol = 1e-9 * np.maximum(1.0, np.abs(exact_d2)) + 1e-12
    return bool(np.all(got >= exact_d2[:, : got.shape[1]] - tol))


def latency_stats(prefix: str, seconds) -> dict[str, float]:
    us = 1e6 * np.asarray(seconds, dtype=float)
    if us.size == 0:
        return {}
    out = {
        f"{prefix}_{name}_us": float(np.percentile(us, p)) for name, p in PERCENTILES
    }
    out[f"{prefix}_max_us"] = float(us.max())
    out[f"{prefix}_mean_us"] = float(us.mean())
    return out


def io_counters() -> dict[str, float]:
    try:
        text = Path("/proc/self/io").read_text(encoding="ascii")
    except OSError:
        return dict.fromkeys(IO_KEYS, float("nan"))
    fields = dict(line.split(": ") for line in text.splitlines() if ": " in line)
    return {key: float(fields.get(key, "nan")) for key in IO_KEYS}


def io_delta(before: dict, after: dict, per: float, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{k}": (after[k] - before[k]) / max(per, 1.0) for k in IO_KEYS}


def work_parent(explicit: str | None = None) -> str:
    path = Path(explicit or os.environ.get(WORK_PARENT_ENV) or DEFAULT_WORK_PARENT)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def fs_type(path: str) -> str:
    best, kind = "", "unknown"
    real = os.path.realpath(path)
    try:
        lines = Path("/proc/mounts").read_text(encoding="ascii").splitlines()
    except OSError:
        return kind
    for line in lines:
        parts = line.split()
        mount = parts[1] if len(parts) > 2 else ""
        inside = real == mount or real.startswith(mount.rstrip("/") + "/")
        if inside and len(mount) > len(best):
            best, kind = mount, parts[2]
    return kind
