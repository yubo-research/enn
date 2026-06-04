"""Full hannoy incremental smoke (used by tests and CLI)."""

from __future__ import annotations

import tempfile
import time

import numpy as np

_SMOKE_SHAPE = (32, 2_000, 500, 10, 64)  # dim, n0, n_add, k, ef


def _timed_writer_add(db, dim: int, ef: int, ids: range, data: np.ndarray) -> float:
    t0 = time.perf_counter()
    with db.writer(dim, m=16, ef=ef) as writer:
        writer.add_items(list(ids), data)
    return time.perf_counter() - t0


def _timed_search(db, query: np.ndarray, k: int, ef: int) -> tuple[float, list]:
    reader = db.reader()
    t0 = time.perf_counter()
    nns = reader.by_vec(query.tolist(), n=k, ef_search=ef)
    return time.perf_counter() - t0, nns


def _report_smoke(tmp: str, m: dict) -> None:
    print(f"db path: {tmp}")
    print(f"vectors: {m['n0']} initial + {m['n_add']} incremental, dim={m['dim']}")
    print(f"build: {m['build_s']:.3f}s  incremental: {m['incr_s']:.3f}s")
    print(
        f"search before add: {m['search0_s'] * 1e3:.2f} ms -> top id {m['nns0'][0][0]}"
    )
    print(
        f"search after add:  {m['search1_s'] * 1e3:.2f} ms -> top id {m['nns1'][0][0]}"
    )
    print(f"recall@{m['k']} vs brute force: {m['recall']:.2f}")
    print(f"top-{m['k']} ids: {[i for i, _ in m['nns1']]}")
    print(f"ground truth: {[i for i, _ in m['gt']]}")


def _smoke_metrics_for_dir(tmp: str) -> dict:
    import hannoy
    from hannoy import Metric

    from examples.try_hannoy import brute_knn

    dim, n0, n_add, k, ef = _SMOKE_SHAPE
    rng = np.random.default_rng(0)
    data0 = rng.standard_normal((n0, dim), dtype=np.float32)
    data_add = rng.standard_normal((n_add, dim), dtype=np.float32)
    query = rng.standard_normal(dim, dtype=np.float32)
    db = hannoy.Database(tmp, Metric.EUCLIDEAN)
    build_s = _timed_writer_add(db, dim, ef, range(n0), data0)
    search0_s, nns0 = _timed_search(db, query, k, ef)
    incr_s = _timed_writer_add(db, dim, ef, range(n0, n0 + n_add), data_add)
    search1_s, nns1 = _timed_search(db, query, k, ef)
    gt = brute_knn(query, np.vstack([data0, data_add]), k)
    recall = len({i for i, _ in nns1} & {i for i, _ in gt}) / k
    return {
        "n0": n0,
        "n_add": n_add,
        "dim": dim,
        "k": k,
        "build_s": build_s,
        "incr_s": incr_s,
        "search0_s": search0_s,
        "search1_s": search1_s,
        "nns0": nns0,
        "nns1": nns1,
        "recall": recall,
        "gt": gt,
    }


def run_hannoy_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="enn_hannoy_") as tmp:
        _report_smoke(tmp, _smoke_metrics_for_dir(tmp))


if __name__ == "__main__":
    run_hannoy_smoke()
