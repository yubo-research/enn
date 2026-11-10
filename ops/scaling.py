#!/usr/bin/env python
"""Measure how ENN add and query times scale with the number of observations N.

Grows one disk-backed ENN through N = 1, 3, 10, 30, ... and, at each N, records
(one JSON object per line on stdout):

- ``add_row_s``       mean seconds per row of the batched adds since the last N
                       (``add`` + ``schedule_background_flush`` only);
- ``add_sync_row_s``  the same, plus a forced ``ensure_index_sync`` at N;
- ``add1_s``          median seconds for ONE single-row ``add`` into an index of N;
- ``query_batch_s``   median seconds for one ``posterior`` call on ``--batch-q``
                       query points;
- ``query1_s``        median seconds for one ``posterior`` call on a single point.
"""

from __future__ import annotations

import json
import resource
import shutil
import sys
import time
from pathlib import Path

import click
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams

K = 10
PARAMS = ENNParams(
    k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
ADD_BATCH = 1000
MIN_PROBE_N = 100


def grid(max_n: int) -> list[int]:
    out, p = [], 1
    while p <= max_n:
        for m in (1, 3):
            if m * p <= max_n:
                out.append(m * p)
        p *= 10
    return out


def _dir_bytes(path: str) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def _rss_split_mib() -> dict[str, float]:
    """Current (not peak) RSS split into anonymous and file-backed MiB."""
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("VmRSS", "RssAnon", "RssFile"):
                out[k] = int(v.split()[0]) / 1024
    return out


def _med_time(fn, reps: int) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run(
    *,
    max_n: int,
    num_dim: int,
    num_metrics: int = 1,
    seed: int,
    work_dir: str,
    batch_q: int,
    reps: int,
    extra_gb: tuple[float, ...] = (),
    stop_rss_mib: float = 0.0,
) -> None:
    row_bytes = 8 * (num_dim + num_metrics)
    extra_n = [int(g * 2**30 / row_bytes) for g in extra_gb]
    targets = sorted({*grid(max_n), *(t for t in extra_n if t <= max_n)})
    rng = np.random.default_rng(seed)
    qrng = np.random.default_rng(seed + 10_000)
    ex, ey = np.empty((0, num_dim)), np.empty((0, num_metrics))
    model = EpistemicNearestNeighbors(ex, ey, work_dir=work_dir)
    model.add(ex, ey)
    n = 0
    for target in targets:
        add_s, rows = 0.0, target - n
        while n < target:
            b = min(ADD_BATCH, target - n)
            x, y = (
                rng.standard_normal((b, num_dim)),
                rng.standard_normal((b, num_metrics)),
            )
            t0 = time.perf_counter()
            model.add(x, y)
            model.schedule_background_flush()
            add_s += time.perf_counter() - t0
            n += b
            if stop_rss_mib and _rss_split_mib()["VmRSS"] > stop_rss_mib:
                print(
                    json.dumps({"aborted_rss_mib": _rss_split_mib(), "n_actual": n}),
                    flush=True,
                )
                return
        t0 = time.perf_counter()
        model.ensure_index_sync()
        sync_s = time.perf_counter() - t0
        qb = qrng.standard_normal((batch_q, num_dim))
        model.posterior(qb, params=PARAMS)  # warm-up
        query_batch_s = _med_time(
            lambda qb=qb: model.posterior(qb, params=PARAMS), reps
        )
        q1 = qrng.standard_normal((reps, 1, num_dim))
        it = iter(q1)
        query1_s = _med_time(
            lambda it=it: model.posterior(next(it), params=PARAMS), reps
        )
        add1_s = None
        if target >= MIN_PROBE_N:
            ts = []
            for _ in range(reps):
                x = rng.standard_normal((1, num_dim))
                y = rng.standard_normal((1, num_metrics))
                t0 = time.perf_counter()
                model.add(x, y)
                model.schedule_background_flush()
                ts.append(time.perf_counter() - t0)
                n += 1
            add1_s = float(np.median(ts))
        rec = {
            "n": target,
            "n_actual": n,
            "seed": seed,
            "num_dim": num_dim,
            "num_metrics": num_metrics,
            "add_row_s": add_s / rows,
            "add_sync_row_s": (add_s + sync_s) / rows,
            "add1_s": add1_s,
            "query_batch_s": query_batch_s,
            "query_batch_q": batch_q,
            "query1_s": query1_s,
            "data_gib": n * row_bytes / 2**30,
            "disk_gib": _dir_bytes(work_dir) / 2**30,
            "rss_mib": _rss_split_mib(),
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        print(json.dumps(rec), flush=True)
    model.persist_index_to_disk()


@click.command()
@click.option("--max-n", type=int, default=1_000_000, show_default=True)
@click.option("--num-dim", type=int, default=10, show_default=True)
@click.option("--num-metrics", type=int, default=1, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--batch-q", type=int, default=1000, show_default=True)
@click.option("--reps", type=int, default=20, show_default=True)
@click.option("--work-dir", required=True, type=click.Path(file_okay=False))
@click.option(
    "--extra-gb",
    default="",
    help="Comma-separated data sizes (GiB) to add to the grid.",
)
@click.option(
    "--stop-rss-mib", type=float, default=0.0, help="Abort when VmRSS exceeds this."
)
@click.option("--ennbo-config", type=click.Path(dir_okay=False), default=None)
def main(
    max_n: int,
    num_dim: int,
    num_metrics: int,
    seed: int,
    batch_q: int,
    reps: int,
    work_dir: str,
    extra_gb: str,
    stop_rss_mib: float,
    ennbo_config: str | None,
) -> None:
    """Emit JSON-lines timings of ENN add/query versus N."""
    if ennbo_config is not None:
        from enn._rust import set_config_path

        set_config_path(ennbo_config)
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        run(
            max_n=max_n,
            num_dim=num_dim,
            num_metrics=num_metrics,
            seed=seed,
            work_dir=work_dir,
            batch_q=batch_q,
            reps=reps,
            extra_gb=tuple(float(g) for g in extra_gb.split(",") if g),
            stop_rss_mib=stop_rss_mib,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
