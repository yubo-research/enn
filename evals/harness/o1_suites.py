from __future__ import annotations

from evals.harness.o1_drift import SCHEDULES, DriftConfig, run_drift
from evals.harness.o1_eval import STORES, O1EvalConfig, run_config
from evals.harness.o1_workloads import KINDS, Workload

SHORT_N = (1000, 3000, 10000)
LONG_N = (1000, 10000, 100000, 1000000, 10000000)
LONG_D = (3, 10, 30, 100, 1000)
MAX_CELLS = 10**9
SUBSPACE_DIMS = (10, 100, 1000)
INTRINSIC = 3


def short_workloads() -> list[Workload]:
    out = [Workload(kind, 10) for kind in KINDS if kind != "subspace"]
    return out + [Workload("subspace", 100, INTRINSIC)]


def long_workloads() -> list[Workload]:
    out = []
    for d in LONG_D:
        out += [Workload(kind, d) for kind in KINDS if kind != "subspace"]
        if d in SUBSPACE_DIMS:
            out.append(Workload("subspace", d, INTRINSIC))
    return out


def n_grid_for(num_dim: int, grid: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(n for n in grid if n * num_dim <= MAX_CELLS)


def run_workload_suite(workloads, grid, insert_batch: int, stores=STORES) -> None:
    for wl in workloads:
        for store in stores:
            config = O1EvalConfig(
                workload=wl,
                store=store,
                n_grid=n_grid_for(wl.num_dim, grid),
                insert_batch=insert_batch,
            )
            run_config(config)


def run_drift_suite(dims, grid, stores=STORES) -> None:
    for d in dims:
        for schedule in SCHEDULES:
            for store in stores:
                wl = Workload("uniform", d)
                run_drift(
                    DriftConfig(
                        schedule=schedule, workload=wl, store=store, n_grid=grid
                    )
                )


def short_workload_suite() -> None:
    run_workload_suite(short_workloads(), SHORT_N, insert_batch=1)


def long_scaling_suite() -> None:
    run_workload_suite(long_workloads(), LONG_N, insert_batch=100)


def short_drift_suite() -> None:
    run_drift_suite((10,), SHORT_N)


def long_drift_suite() -> None:
    run_drift_suite((10, 100), (1000, 3000, 10000, 30000, 100000, 300000))
