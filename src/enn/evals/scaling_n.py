from __future__ import annotations

import math
import multiprocessing
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from enn.enn.enn_fitter import ENNStatefulFitter
from enn.evals.metric_12d import (
    BATCH,
    NUM_FIT_CANDIDATES,
    NUM_FIT_SAMPLES,
    NUM_TEST,
    K,
    build_model,
    make_data,
)
from enn.evals.metrics import format_directed, format_plain, gaussian_loglik, rmse
from enn.evals.scaling_fit import ALPHA, RegFit, best_term, single_term_fits
from ops.stress import DRAW_FLAGS, MeanSE, format_mean_se, mean_se

MODEL = "bpann_disk_auto"
MODEL_LABEL = "bpann_disk_auto_ols"
N_GRID: tuple[int, ...] = (100, 300, 1000, 3000, 10000, 30000, 100000)
MIB = 1024.0 * 1024.0
WORK_DIR_PREFIX = "enn_scaling_n_"
MEMORY_METRICS: tuple[str, ...] = (
    "rss_mib",
    "anon_mib",
    "file_mib",
    "peak_mib",
    "disk_mib",
)
METRICS: tuple[str, ...] = (
    *MEMORY_METRICS,
    "add_s",
    "add_us",
    "fit_s",
    "query_s",
    "query_us",
    "loglik",
    "nrmse",
)
LARGER_METRICS = frozenset({"loglik"})
REG_METRICS: tuple[str, ...] = (*MEMORY_METRICS, "add_us", "fit_s", "query_us")


@dataclass(frozen=True)
class ScalingConfig:
    n_grid: tuple[int, ...] = N_GRID
    num_test: int = NUM_TEST
    batch: int = BATCH
    k: int = K
    num_fit_candidates: int = NUM_FIT_CANDIDATES
    num_fit_samples: int = NUM_FIT_SAMPLES
    seed: int = 0
    num_seeds: int = 3
    alpha: float = ALPHA
    isolate: bool = True


def _proc_status_kib(key: str) -> float:
    try:
        text = Path("/proc/self/status").read_text(encoding="ascii")
    except OSError:
        return math.nan
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return float(line.split()[1])
    return math.nan


def memory_mib() -> dict[str, float]:
    keys = {
        "rss_mib": "VmRSS",
        "anon_mib": "RssAnon",
        "file_mib": "RssFile",
        "peak_mib": "VmHWM",
    }
    return {name: _proc_status_kib(key) / 1024.0 for name, key in keys.items()}


def reset_peak() -> None:
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as f:
            f.write("5")
    except OSError:
        pass


def dir_mib(path: str) -> float:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / MIB


class ScalingRun:
    def __init__(self, work_dir: str, config: ScalingConfig) -> None:
        self.work_dir = work_dir
        self.config = config
        self.model = None
        self.fitter = ENNStatefulFitter(
            k=config.k, rng=np.random.default_rng(config.seed + 1)
        )
        self.params = None
        self.base_mem = memory_mib()

    def memory_delta(self) -> dict[str, float]:
        now = memory_mib()
        out = {
            k: now[k] - self.base_mem[k] for k in ("rss_mib", "anon_mib", "file_mib")
        }
        out["peak_mib"] = now["peak_mib"] - self.base_mem["rss_mib"]
        return out

    def add(self, x: np.ndarray, y: np.ndarray) -> float:
        t0 = time.perf_counter()
        for start in range(0, len(x), self.config.batch):
            xb, yb = (
                x[start : start + self.config.batch],
                y[start : start + self.config.batch],
            )
            if self.model is None:
                self.model = build_model(MODEL, xb, yb, self.work_dir)
            else:
                self.model.add(xb, yb)
            self.model.ensure_index_sync()
            y_bounds = np.asarray(self.model.rust_backend.y_bounds, dtype=float)
            self.fitter.tell(xb, yb, y_bounds=y_bounds)
        return time.perf_counter() - t0

    def fit(self) -> float:
        t0 = time.perf_counter()
        self.params = self.fitter.ask(
            self.model,
            num_fit_candidates=self.config.num_fit_candidates,
            num_fit_samples=self.config.num_fit_samples,
            affine_calibrate=True,
        )
        return time.perf_counter() - t0

    def query(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        post = self.fitter.posterior(self.model, x_test, self.params, flags=DRAW_FLAGS)
        return post.mu, post.se, time.perf_counter() - t0


def run_seed(config: ScalingConfig) -> list[dict[str, float]]:
    rng = np.random.default_rng(config.seed)
    x, y = make_data(max(config.n_grid), rng)
    x_test, y_test = make_data(config.num_test, rng)
    rows: list[dict[str, float]] = []
    with tempfile.TemporaryDirectory(prefix=WORK_DIR_PREFIX) as work_dir:
        run = ScalingRun(work_dir, config)
        lo = 0
        for hi in config.n_grid:
            reset_peak()
            add_s = run.add(x[lo:hi], y[lo:hi])
            fit_s = run.fit()
            mu, se, query_s = run.query(x_test)
            row = {
                "n": float(hi),
                **run.memory_delta(),
                "disk_mib": dir_mib(work_dir),
                "add_s": add_s,
                "add_us": 1e6 * add_s / (hi - lo),
                "fit_s": fit_s,
                "query_s": query_s,
                "query_us": 1e6 * query_s / config.num_test,
                "loglik": gaussian_loglik(y_test, mu, se),
                "nrmse": rmse(y_test, mu) / float(np.std(y_test)),
            }
            print(format_seed_line(config.seed, row), flush=True)
            rows.append(row)
            lo = hi
    return rows


def run_seed_isolated(config: ScalingConfig) -> list[dict[str, float]]:
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(run_seed, config).result()


def format_seed_line(seed: int, row: dict[str, float]) -> str:
    vals = " ".join(format_plain(k, f"{v:.4f}") for k, v in row.items() if k != "n")
    return f"seed = {seed} n = {int(row['n'])} {vals}"


def summarize(rows: list[dict[str, float]]) -> dict[int, dict[str, MeanSE]]:
    groups: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        groups.setdefault(int(row["n"]), []).append(row)
    return {
        n: {k: mean_se([r[k] for r in group]) for k in group[0] if k != "n"}
        for n, group in sorted(groups.items())
    }


def format_eval_line(n: int, stats: dict[str, MeanSE]) -> str:
    vals = " ".join(
        format_directed(
            "LARGER" if k in LARGER_METRICS else "SMALLER",
            k,
            format_mean_se(stats[k], fmt=".4f"),
        )
        for k in METRICS
    )
    return f"EVAL: {format_plain('model', MODEL_LABEL)} {format_plain('n', n)} {vals}"


def format_reg_line(metric: str, fits: dict[str, RegFit], alpha: float = ALPHA) -> str:
    obs = next(iter(fits.values())).obs if fits else 0
    parts = [
        format_plain("model", MODEL_LABEL),
        format_plain("reg", metric),
        format_plain("obs", obs),
        format_plain("best", best_term(fits, alpha)),
    ]
    for term, fit in fits.items():
        tt = fit.tests[term]
        parts += [
            format_plain(f"r2_{term}", f"{fit.r2:.4f}"),
            format_plain(f"b0_{term}", f"{fit.intercept:.4g}"),
            format_plain(f"b_{term}", f"{tt.coef:.4g}"),
            format_plain(f"t_{term}", f"{tt.t:.3g}"),
            format_plain(f"p_{term}", f"{tt.p:.3g}"),
        ]
    return "EVAL: " + " ".join(parts)


def regress(rows: list[dict[str, float]]) -> dict[str, dict[str, RegFit]]:
    ns = [r["n"] for r in rows]
    return {m: single_term_fits(ns, [r[m] for r in rows]) for m in REG_METRICS}


def run_eval(
    config: ScalingConfig | None = None,
) -> tuple[dict[int, dict[str, MeanSE]], dict[str, dict[str, RegFit]]]:
    cfg = ScalingConfig() if config is None else config
    if list(cfg.n_grid) != sorted(set(cfg.n_grid)) or cfg.n_grid[0] < 2:
        raise ValueError("n_grid must be strictly increasing and start at >= 2")
    if len(cfg.n_grid) < 3:
        raise ValueError("comparing ln N, N and N^2 fits needs >= 3 checkpoints")
    print(
        f"model={MODEL_LABEL} n_grid={','.join(map(str, cfg.n_grid))} num_test={cfg.num_test} "
        f"batch={cfg.batch} seeds={cfg.seed}..{cfg.seed + cfg.num_seeds - 1} alpha={cfg.alpha}",
        flush=True,
    )
    rows: list[dict[str, float]] = []
    for i in range(cfg.num_seeds):
        seed_cfg = replace(cfg, seed=cfg.seed + i)
        rows.extend(run_seed_isolated(seed_cfg) if cfg.isolate else run_seed(seed_cfg))
    summary = summarize(rows)
    for n, stats in summary.items():
        print(format_eval_line(n, stats), flush=True)
    fits = regress(rows)
    for metric, term_fits in fits.items():
        print(format_reg_line(metric, term_fits, cfg.alpha), flush=True)
    return summary, fits
