from __future__ import annotations

import json
import multiprocessing
import sys
import time
from dataclasses import asdict, dataclass, replace

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fitter import ENNStatefulFitter
from enn.enn.enn_store import ENNStore
from evals.harness.metrics import gaussian_loglik, rmse
from evals.harness.scaling_n import ScalingRun, dir_mib, memory_mib, reset_peak
from evals.harness.work_dirs import scoped_work_dir
from ops.stress import DRAW_FLAGS

NOISE_STD = 0.1
NUM_RELEVANT = 5
N_GRID: tuple[int, ...] = (1000, 3000, 10000, 30000, 100000)
DIM_GRID: tuple[int, ...] = (10, 100, 1000)
NUM_DRAW_SEEDS = 10


@dataclass(frozen=True)
class StoreCompareConfig:
    store: str = "O1"
    num_dim: int = 10
    n_grid: tuple[int, ...] = N_GRID
    num_test: int = 300
    batch: int = 1000
    k: int = 10
    num_fit_candidates: int = 30
    num_fit_samples: int = 100
    seed: int = 0
    work_parent: str | None = None


def make_data(
    num_obs: int, num_dim: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.random((num_obs, num_dim))
    rel = min(NUM_RELEVANT, num_dim)
    weights = 1.0 / np.arange(1, rel + 1)
    f = np.sin(2 * np.pi * x[:, :rel]) @ weights
    y = f + NOISE_STD * rng.standard_normal(num_obs)
    return x, y.reshape(-1, 1)


class StoreRun(ScalingRun):
    def __init__(self, work_dir: str, config: StoreCompareConfig) -> None:
        self.work_dir = work_dir
        self.config = config
        self.model: EpistemicNearestNeighbors | None = None
        self.fitter = ENNStatefulFitter(k=config.k, seed=config.seed + 1)
        self.params = None
        self.base = memory_mib()

    def add(self, num_rows: int, rng: np.random.Generator) -> float:
        elapsed = 0.0
        for start in range(0, num_rows, self.config.batch):
            x, y = make_data(
                min(self.config.batch, num_rows - start), self.config.num_dim, rng
            )
            t0 = time.perf_counter()
            if self.model is None:
                self.model = EpistemicNearestNeighbors(
                    x,
                    y,
                    work_dir=self.work_dir,
                    weighting=True,
                    store=self.config.store,
                )
            else:
                self.model.add(x, y)
            self.model.ensure_index_sync()
            elapsed += time.perf_counter() - t0
        return elapsed

    def query(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        t0 = time.perf_counter()
        post = self.model.posterior(x_test, params=self.params, flags=DRAW_FLAGS)
        t1 = time.perf_counter()
        self.model.posterior_function_draw(
            x_test,
            params=self.params,
            function_seeds=np.arange(NUM_DRAW_SEEDS),
            flags=DRAW_FLAGS,
        )
        return post.mu, post.se, t1 - t0, time.perf_counter() - t1

    def memory(self) -> dict[str, float]:
        now = memory_mib()
        out = {k: now[k] - self.base[k] for k in ("rss_mib", "anon_mib", "file_mib")}
        out["peak_mib"] = now["peak_mib"] - self.base["rss_mib"]
        out["disk_mib"] = dir_mib(self.work_dir)
        return out


def calibration_fields(run: StoreRun, test) -> dict:
    x_test, y_test = test
    cal = run.params.calibrator
    if cal is None:
        return {"calibrated": False}
    raw = run.model.posterior(x_test, params=replace(run.params, calibrator=None))
    return {
        "calibrated": True,
        "cal_a": float(cal.a[0]),
        "cal_b": float(cal.b[0]),
        "cal_c": float(cal.c[0]),
        "loglik_raw": gaussian_loglik(y_test, raw.mu, raw.se),
        "nrmse_raw": rmse(y_test, raw.mu) / float(np.std(y_test)),
    }


def checkpoint_row(run: StoreRun, n: int, timings: dict, test) -> dict:
    x_test, y_test = test
    mu, se, post_s, draw_s = run.query(x_test)
    num_test = len(y_test)
    return {
        **calibration_fields(run, test),
        **asdict(run.config),
        "n": n,
        **run.memory(),
        **timings,
        "post_us": 1e6 * post_s / num_test,
        "draw_us": 1e6 * draw_s / num_test,
        "loglik": gaussian_loglik(y_test, mu, se),
        "nrmse": rmse(y_test, mu) / float(np.std(y_test)),
        "k_fit": int(run.params.k_num_neighbors),
    }


def run_config(config: StoreCompareConfig) -> list[dict]:
    rng = np.random.default_rng(config.seed)
    test = make_data(config.num_test, config.num_dim, rng)
    rows = []
    with scoped_work_dir("enn_store_cmp_", config.work_parent) as work_dir:
        run = StoreRun(work_dir, config)
        lo = 0
        for hi in config.n_grid:
            reset_peak()
            add_s = run.add(hi - lo, rng)
            timings = {"add_us": 1e6 * add_s / (hi - lo), "fit_s": run.fit()}
            rows.append(checkpoint_row(run, hi, timings, test))
            print(json.dumps(rows[-1]), flush=True)
            lo = hi
    return rows


def run_isolated(config: StoreCompareConfig) -> list[dict]:
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(run_config, (config,))


def main(argv: list[str]) -> None:
    out_path, work_parent = argv[0], argv[1]
    dims = tuple(int(d) for d in argv[2].split(",")) if len(argv) > 2 else DIM_GRID
    stores = argv[3].split(",") if len(argv) > 3 else [s.value for s in ENNStore]
    num_seeds = int(argv[4]) if len(argv) > 4 else 3
    for seed in range(num_seeds):
        run_grid(out_path, work_parent, dims, stores, seed)


def run_grid(out_path: str, work_parent: str, dims, stores, seed: int) -> None:
    for num_dim in dims:
        for store in stores:
            config = StoreCompareConfig(
                store=store, num_dim=num_dim, work_parent=work_parent, seed=seed
            )
            rows = run_isolated(config)
            with open(out_path, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(row) + "\n" for row in rows)


if __name__ == "__main__":
    main(sys.argv[1:])
