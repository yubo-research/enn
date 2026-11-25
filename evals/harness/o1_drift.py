from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from enn import _rust
from evals.harness import o1_measure as ms
from evals.harness.metrics import format_larger, format_smaller
from evals.harness.o1_eval import O1EvalConfig, O1EvalRun, write_row
from evals.harness.o1_workloads import Workload, queries
from evals.harness.work_dirs import scoped_work_dir

SCHEDULES = ("gradual", "abrupt", "relevant_dims", "alternating")
DRIFT_RANGE = 10.0
RELEVANT_SCALE = 0.1
FROZEN_LEARNER = 1e12
STALE_DRIFT = 0.25
SIDES = (("inc", False, None), ("ref", True, 0), ("ada", True, 30))


@dataclass(frozen=True)
class DriftConfig:
    schedule: str = "gradual"
    workload: Workload = field(default_factory=Workload)
    store: str = "O1"
    n_grid: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000)
    num_queries: int = 100
    k: int = 10
    insert_batch: int = 100
    work_parent: str | None = None


def _relevant(num_dim: int, first: bool) -> np.ndarray:
    scale = np.ones(num_dim)
    half = max(1, num_dim // 2)
    scale[slice(0, half) if first else slice(half, None)] = RELEVANT_SCALE
    return scale


def metric_scale(schedule: str, step: int, num_steps: int, num_dim: int) -> np.ndarray:
    t = step / max(1, num_steps - 1)
    tilt = np.linspace(-1.0, 1.0, num_dim) * np.log(DRIFT_RANGE)
    if schedule == "gradual":
        return np.exp(t * tilt)
    if schedule == "abrupt":
        return np.exp(tilt) if t >= 0.5 else np.ones(num_dim)
    if schedule == "relevant_dims":
        return _relevant(num_dim, t < 0.5)
    if schedule == "alternating":
        return _relevant(num_dim, step % 2 == 0)
    raise ValueError(f"unknown drift schedule: {schedule!r}")


class DriftRun(O1EvalRun):
    def __init__(
        self, work_dir: str, config: O1EvalConfig, repair_rows: int | None = None
    ) -> None:
        super().__init__(work_dir, config)
        self.repair_rows = repair_rows

    def _build(self, hi: int) -> None:
        super()._build(hi)
        backend = self.model.rust_backend
        _rust.metric_configure(
            backend, FROZEN_LEARNER, FROZEN_LEARNER, None, None, None
        )
        if self.repair_rows is not None and self.config.store == "O1F":
            _rust.index_set_repair(backend, self.repair_rows, STALE_DRIFT, True)

    def install(self, scale: np.ndarray, rebuild: bool) -> dict[str, float]:
        io0, t0 = ms.io_counters(), time.perf_counter()
        self.model.rust_backend.set_metric_scale(scale, rebuild)
        out = {"set_metric_s": time.perf_counter() - t0}
        out.update(ms.io_delta(io0, ms.io_counters(), 1.0, "set_metric_io"))
        return out

    def recall_under(self, scale: np.ndarray, q: np.ndarray, exact) -> dict:
        backend = self.model.rust_backend
        d2, idx = backend.neighbor_distances_and_indices(q, self.config.k)
        out = {
            "recall": ms.recall(exact[1], idx),
            "metric_ok": ms.metric_consistent(exact[0], d2),
        }
        occ = _rust.index_occupancy(self.model.rust_backend)
        if occ is not None:
            out["reachable_frac"] = occ["reachable_rows"] / self.num_rows
        return out


def _side(run: DriftRun, scale, rebuild: bool, q, exact, hi: int) -> dict:
    out = run.insert(hi)
    out.update(run.install(scale, rebuild))
    out.update(run.recall_under(scale, q, exact))
    return out


def drift_step(runs: dict, config: DriftConfig, step: int, hi: int) -> dict:
    wl = config.workload
    scale = metric_scale(config.schedule, step, len(config.n_grid), wl.num_dim)
    q, _ = queries(wl, hi, config.num_queries, "data", step)
    exact = ms.exact_knn(wl, hi, q, 1.0 / scale**2, config.k)
    row = {**asdict(config), "workload": wl.label(), "step": step, "n": hi}
    row.pop("work_parent")
    for name, rebuild, _ in SIDES:
        if name not in runs:
            continue
        side = _side(runs[name], scale, rebuild, q, exact, hi)
        row.update({f"{name}_{k}": v for k, v in side.items()})
    row["scale_ratio"] = float(scale.max() / scale.min())
    row["work_fs"] = ms.fs_type(runs["ref"].work_dir)
    return row


def drift_summary(row: dict) -> str:
    return " ".join(
        [
            f"schedule={row['schedule']} store={row['store']} step={row['step']} n={row['n']}",
            format_larger("inc_recall", f"{row['inc_recall']:.3f}"),
            format_larger("ref_recall", f"{row['ref_recall']:.3f}"),
            format_smaller("ref_rebuild_s", f"{row['ref_set_metric_s']:.4f}"),
            *(
                [format_larger("ada_recall", f"{row['ada_recall']:.3f}")]
                if "ada_recall" in row
                else []
            ),
        ]
    )


def run_drift(config: DriftConfig, out_path: str | None = None) -> list[dict]:
    base = O1EvalConfig(
        workload=config.workload,
        store=config.store,
        k=config.k,
        insert_batch=config.insert_batch,
        predict=False,
    )
    results = []
    parent = ms.work_parent(config.work_parent)
    sides = [s for s in SIDES if s[0] != "ada" or config.store == "O1F"]
    with contextlib.ExitStack() as stack:
        runs = {
            name: DriftRun(
                stack.enter_context(scoped_work_dir(f"enn_o1_drift_{name}_", parent)),
                base,
                repair,
            )
            for name, _, repair in sides
        }
        for step, hi in enumerate(config.n_grid):
            results.append(drift_step(runs, config, step, hi))
            print(drift_summary(results[-1]), flush=True)
            write_row(results[-1], out_path)
    return results
