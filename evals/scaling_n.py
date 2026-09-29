"""Scaling with N of memory, add time and query time for BPANN_DISK + AUTO + OLS.

One model: ``index_driver=BPANN_DISK``, ``metric_learning=AUTO`` and the post-hoc OLS affine
calibrator (``ENNStatefulFitter.ask(..., affine_calibrate=True)``). Rows of the 12-d
``metric_12d`` problem stream in batches with ``ensure_index_sync`` after each batch; the
fitter is told only the new rows, so it holds running moments of ``y`` rather than the rows.
At each checkpoint in ``n_grid`` the model refits its hyperparameters and calibrator, then
answers one posterior call on ``num_test`` fixed test points.

Per checkpoint:

- memory: ``rss_mib`` is resident memory after the checkpoint minus resident memory before the
  model was built, split into ``anon_mib`` (heap) and ``file_mib`` (pages of memory-mapped
  files, which the kernel can reclaim); ``peak_mib`` is the peak resident memory since the
  previous checkpoint (``VmHWM`` reset through ``/proc/self/clear_refs``) minus the same
  baseline; ``disk_mib`` is the size of the model's work directory. Linux only (nan elsewhere).
- add: ``add_s`` is the wall time of the adds and syncs since the previous checkpoint (including
  AUTO's metric refits inside ``add``), ``add_us`` that time per added row, and ``fit_s`` the
  hyperparameter and calibrator fit.
- query: ``query_s`` is the posterior call and ``query_us`` that time per test point.
- accuracy on the test points: ``loglik`` and ``nrmse`` (RMSE over the std of ``y_test``).

Each seed runs in its own process. Each checkpoint line gives mean ± standard error over seeds.
For each metric in ``REG_METRICS`` two ``reg`` lines follow (see ``evals.scaling_fit``): the OLS
fit of the per-seed values on ln N, N and N^2 (N in thousands) with each term's t and p value,
and the fit left after backward elimination at ``alpha``, naming accepted and rejected terms.
"""

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
from evals.flat_sphere import gaussian_loglik, rmse
from evals.metric_12d import (
    BATCH,
    K,
    NUM_FIT_CANDIDATES,
    NUM_FIT_SAMPLES,
    NUM_TEST,
    build_model,
    make_data,
)
from evals.scaling_fit import ALPHA, TERMS, RegFit, backward_eliminate
from evals.stress_eval import format_directed, format_plain
from ops.stress import DRAW_FLAGS, MeanSE, format_mean_se, mean_se

MODEL = "bpann_disk_auto"
MODEL_LABEL = "bpann_disk_auto_ols"
N_GRID: tuple[int, ...] = (100, 300, 1000, 3000, 10000, 30000, 100000)
MIB = 1024.0 * 1024.0
WORK_DIR_PREFIX = "enn_scaling_n_"
MEMORY_METRICS: tuple[str, ...] = ("rss_mib", "anon_mib", "file_mib", "peak_mib", "disk_mib")
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
    """Resident (total, anonymous, file-backed) and peak resident memory in MiB."""
    keys = {"rss_mib": "VmRSS", "anon_mib": "RssAnon", "file_mib": "RssFile", "peak_mib": "VmHWM"}
    return {name: _proc_status_kib(key) / 1024.0 for name, key in keys.items()}


def reset_peak() -> None:
    """Reset ``VmHWM`` to the current resident size (Linux >= 4.0; no-op elsewhere)."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as f:
            f.write("5")
    except OSError:
        pass


def dir_mib(path: str) -> float:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / MIB


class ScalingRun:
    """One seed's model, grown checkpoint to checkpoint."""

    def __init__(self, work_dir: str, config: ScalingConfig) -> None:
        self.work_dir = work_dir
        self.config = config
        self.model = None
        self.fitter = ENNStatefulFitter(k=config.k, rng=np.random.default_rng(config.seed + 1))
        self.params = None
        self.base_mem = memory_mib()

    def memory_delta(self) -> dict[str, float]:
        """Memory now minus memory before the model was built; peak is relative to base RSS."""
        now = memory_mib()
        out = {k: now[k] - self.base_mem[k] for k in ("rss_mib", "anon_mib", "file_mib")}
        out["peak_mib"] = now["peak_mib"] - self.base_mem["rss_mib"]
        return out

    def add(self, x: np.ndarray, y: np.ndarray) -> float:
        t0 = time.perf_counter()
        for start in range(0, len(x), self.config.batch):
            xb, yb = x[start : start + self.config.batch], y[start : start + self.config.batch]
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
    """Stream one seed; print and return one row of metrics per checkpoint."""
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
    """``run_seed`` in a fresh process, so no seed inherits another's heap or mapped pages."""
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(run_seed, config).result()


def format_seed_line(seed: int, row: dict[str, float]) -> str:
    """Per-seed progress line with every metric; deliberately not an EVAL line."""
    vals = " ".join(format_plain(k, f"{v:.4f}") for k, v in row.items() if k != "n")
    return f"seed = {seed} n = {int(row['n'])} {vals}"


def summarize(rows: list[dict[str, float]]) -> dict[int, dict[str, MeanSE]]:
    """Mean ± SE over seeds of every metric, keyed by checkpoint n."""
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
            "LARGER" if k in LARGER_METRICS else "SMALLER", k, format_mean_se(stats[k], fmt=".4f")
        )
        for k in METRICS
    )
    return f"EVAL: {format_plain('model', MODEL_LABEL)} {format_plain('n', n)} {vals}"


def _term_names(terms: object) -> str:
    return ",".join(terms) or "none"


def format_reg_line(metric: str, label: str, fit: RegFit) -> str:
    """One ``reg`` line: intercept, then coefficient, t and p value of each term in ``fit``."""
    head = [
        format_plain("model", MODEL_LABEL),
        format_plain("reg", metric),
        format_plain("fit", label),
        format_plain("obs", fit.obs),
        format_plain("r2", f"{fit.r2:.4f}"),
    ]
    if label != "full":
        head += [
            format_plain("accepted", _term_names(fit.tests)),
            format_plain("rejected", _term_names(t for t in TERMS if t not in fit.tests)),
        ]
    head.append(format_plain("b0", f"{fit.intercept:.4g}"))
    for term, tt in fit.tests.items():
        head += [
            format_plain(f"b_{term}", f"{tt.coef:.4g}"),
            format_plain(f"t_{term}", f"{tt.t:.3g}"),
            format_plain(f"p_{term}", f"{tt.p:.3g}"),
        ]
    return "EVAL: " + " ".join(head)


def regress(rows: list[dict[str, float]], alpha: float) -> dict[str, tuple[RegFit, RegFit]]:
    """Full and backward-eliminated fits of each ``REG_METRICS`` metric on per-seed rows."""
    ns = [r["n"] for r in rows]
    return {m: backward_eliminate(ns, [r[m] for r in rows], alpha) for m in REG_METRICS}


def run_eval(
    config: ScalingConfig | None = None,
) -> tuple[dict[int, dict[str, MeanSE]], dict[str, tuple[RegFit, RegFit]]]:
    """Run all seeds; print per-checkpoint EVAL lines and the regression lines."""
    cfg = ScalingConfig() if config is None else config
    if list(cfg.n_grid) != sorted(set(cfg.n_grid)) or cfg.n_grid[0] < 2:
        raise ValueError("n_grid must be strictly increasing and start at >= 2")
    if len(cfg.n_grid) < len(TERMS) + 1 or len(cfg.n_grid) * cfg.num_seeds < len(TERMS) + 2:
        raise ValueError(
            f"regression on {len(TERMS)} terms needs >= {len(TERMS) + 1} checkpoints "
            f"and >= {len(TERMS) + 2} rows (checkpoints x seeds)"
        )
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
    fits = regress(rows, cfg.alpha)
    for metric, (full, reduced) in fits.items():
        print(format_reg_line(metric, "full", full), flush=True)
        print(format_reg_line(metric, "reduced", reduced), flush=True)
    return summary, fits
