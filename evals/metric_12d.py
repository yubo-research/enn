"""Streaming eval on the 12-d "two of twelve" problem, where metric learning matters.

x ~ U[0,1]^12, y = sin(6 pi x0) + sin(6 pi x1) + 0.1 eps, so ten inputs are irrelevant
(reports/mbpann/metric_advantage.py). Rows stream in batches with ``ensure_index_sync``
after each batch. At every checkpoint each model refits its ENN hyperparameters with
``enn_fit``; ``bpann_disk_metric_learning`` first refits diagonal metric weights by LOOCV
(reports/iaml/iaml_core.fit_exact) on a subsample and applies them with
``MBPANNMetric.set_weights``. Once the previous checkpoint already filled the fit
subsample, the metric fit is warm-started from the previous checkpoint's metric
(``metric_warm_start``); earlier fits start cold, since a warm start from a fit on a
few dozen rows stays stuck at the identity-metric level.

``add_s`` is wall time from the previous checkpoint to this one: adds, syncs, metric
fit, and hyperparameter fit. ``query_s`` is one posterior call on the test set.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import tempfile
import time
from dataclasses import dataclass
from types import ModuleType

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.mbpann import MBPANNMetric
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_x_scaling import ENNXScaling
from evals.flat_sphere import gaussian_loglik, rmse
from evals.stress_eval import REPO_ROOT, format_larger, format_plain, format_smaller
from ops.stress import DRAW_FLAGS

NUM_DIM = 12
NOISE_STD = 0.1
N_GRID: tuple[int, ...] = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000)
NUM_TEST = 1000
BATCH = 500
K = 10
NUM_FIT_CANDIDATES = 100
NUM_FIT_SAMPLES = 100
METRIC_FIT_SUBSAMPLE = 1000
SEED = 0
MODELS: tuple[str, ...] = (
    "flat",
    "flat_scale_x",
    "bpann_disk",
    "bpann_disk_metric_learning",
)
BPANN_X_SCALING: dict[str, ENNXScaling] = {
    "bpann_disk": ENNXScaling.NONE,
    "bpann_disk_metric_learning": ENNXScaling.METRIC_LEARNING,
}
WORK_DIR_PREFIX = "enn_metric_12d_"
IAML_CORE_PATH = REPO_ROOT / "reports" / "iaml" / "iaml_core.py"


@dataclass(frozen=True)
class Metric12dConfig:
    n_grid: tuple[int, ...] = N_GRID
    num_test: int = NUM_TEST
    batch: int = BATCH
    k: int = K
    num_fit_candidates: int = NUM_FIT_CANDIDATES
    num_fit_samples: int = NUM_FIT_SAMPLES
    metric_fit_subsample: int = METRIC_FIT_SUBSAMPLE
    metric_warm_start: bool = True
    seed: int = SEED


@dataclass(frozen=True)
class CheckpointResult:
    model: str
    num_obs: int
    loglik: float
    nrmse: float
    add_s: float
    query_s: float


@functools.cache
def load_iaml_core() -> ModuleType:
    spec = importlib.util.spec_from_file_location("iaml_core", IAML_CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {IAML_CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_data(num_obs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.random((num_obs, NUM_DIM))
    f = np.sin(6 * np.pi * x[:, 0]) + np.sin(6 * np.pi * x[:, 1])
    y = f + NOISE_STD * rng.standard_normal(num_obs)
    return x, y.reshape(-1, 1)


def build_model(
    name: str, x: np.ndarray, y: np.ndarray, work_dir: str
) -> EpistemicNearestNeighbors:
    if name == "flat":
        return EpistemicNearestNeighbors(x, y)
    if name == "flat_scale_x":
        return EpistemicNearestNeighbors(x, y, x_scaling=ENNXScaling.SCALE_X)
    if name not in BPANN_X_SCALING:
        raise ValueError(f"unknown model {name!r}")
    model_dir = os.path.join(work_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    return EpistemicNearestNeighbors(
        x,
        y,
        x_scaling=BPANN_X_SCALING[name],
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=model_dir,
    )


def fit_metric(
    helper: MBPANNMetric,
    x: np.ndarray,
    y: np.ndarray,
    config: Metric12dConfig,
    rng: np.random.Generator,
    prev_theta: np.ndarray | None = None,
) -> np.ndarray:
    """LOOCV-fit diagonal metric weights on a subsample, apply them to the index, return theta.

    With ``prev_theta`` the fit is warm-started there: one outer round, no isotropic restart.
    """
    iaml_core = load_iaml_core()
    num_obs = len(x)
    sub = rng.choice(num_obs, size=min(num_obs, config.metric_fit_subsample), replace=False)
    metric = iaml_core.Metric(NUM_DIM)
    k = min(config.k, len(sub) - 1)
    if prev_theta is None:
        iaml_core.fit_exact(metric, x[sub], y[sub, 0], k)
    else:
        metric.theta = prev_theta.copy()
        iaml_core.fit_exact(metric, x[sub], y[sub, 0], k, outer=1, restart=False)
    helper.set_weights(metric.a)
    return metric.theta.copy()


class StreamedModel:
    """One model grown checkpoint to checkpoint on a shared row stream."""

    def __init__(self, name: str, work_dir: str, config: Metric12dConfig) -> None:
        self.name = name
        self.work_dir = work_dir
        self.config = config
        self.model: EpistemicNearestNeighbors | None = None
        self.helper: MBPANNMetric | None = None
        self.params: object = None
        self.metric_theta: np.ndarray | None = None
        self.fit_rng = np.random.default_rng(config.seed + 1)
        if BPANN_X_SCALING.get(name) == ENNXScaling.METRIC_LEARNING:
            load_iaml_core()

    def _add_rows(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.model is None:
            self.model = build_model(self.name, x, y, self.work_dir)
            if self.model.x_scaling == ENNXScaling.METRIC_LEARNING:
                self.helper = MBPANNMetric(self.model)
        else:
            self.model.add(x, y)
        self.model.ensure_index_sync()

    def advance(self, x: np.ndarray, y: np.ndarray, lo: int, hi: int) -> float:
        """Stream rows ``lo:hi`` in batches, refit; return elapsed seconds."""
        t0 = time.perf_counter()
        for start in range(lo, hi, self.config.batch):
            stop = min(hi, start + self.config.batch)
            self._add_rows(x[start:stop], y[start:stop])
        assert self.model is not None
        if self.helper is not None:
            warm = self.config.metric_warm_start and lo >= self.config.metric_fit_subsample
            prev = self.metric_theta if warm else None
            self.metric_theta = fit_metric(
                self.helper, x[:hi], y[:hi], self.config, self.fit_rng, prev
            )
        self.params = enn_fit(
            self.model,
            k=self.config.k,
            num_fit_candidates=self.config.num_fit_candidates,
            num_fit_samples=self.config.num_fit_samples,
            rng=self.fit_rng,
        )
        return time.perf_counter() - t0

    def query(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        assert self.model is not None
        t0 = time.perf_counter()
        post = self.model.posterior(x_test, params=self.params, flags=DRAW_FLAGS)
        return post.mu, post.se, time.perf_counter() - t0


def format_eval_line(result: CheckpointResult) -> str:
    return (
        "EVAL: "
        f"{format_plain('model', result.model)} "
        f"{format_plain('n', result.num_obs)} "
        f"{format_larger('loglik', f'{result.loglik:.4f}')} "
        f"{format_smaller('nrmse', f'{result.nrmse:.4f}')} "
        f"{format_smaller('add_s', f'{result.add_s:.4f}')} "
        f"{format_smaller('query_s', f'{result.query_s:.4f}')}"
    )


def run_model(name: str, work_dir: str, config: Metric12dConfig) -> list[CheckpointResult]:
    """Stream the shared data through one model; print and return one row per checkpoint."""
    rng = np.random.default_rng(config.seed)
    x, y = make_data(max(config.n_grid), rng)
    x_test, y_test = make_data(config.num_test, rng)
    y_test_std = float(np.std(y_test))
    streamed = StreamedModel(name, work_dir, config)
    results: list[CheckpointResult] = []
    lo = 0
    for hi in config.n_grid:
        add_s = streamed.advance(x, y, lo, hi)
        mu, se, query_s = streamed.query(x_test)
        result = CheckpointResult(
            model=name,
            num_obs=hi,
            loglik=gaussian_loglik(y_test, mu, se),
            nrmse=rmse(y_test, mu) / y_test_std,
            add_s=add_s,
            query_s=query_s,
        )
        print(format_eval_line(result), flush=True)
        results.append(result)
        lo = hi
    return results


def run_eval(
    config: Metric12dConfig | None = None, models: tuple[str, ...] = MODELS
) -> list[CheckpointResult]:
    cfg = Metric12dConfig() if config is None else config
    if list(cfg.n_grid) != sorted(set(cfg.n_grid)) or cfg.n_grid[0] < 2:
        raise ValueError("n_grid must be strictly increasing and start at >= 2")
    print(
        f"num_dim={NUM_DIM} num_test={cfg.num_test} batch={cfg.batch} k={cfg.k} "
        f"seed={cfg.seed} models={','.join(models)}",
        flush=True,
    )
    results: list[CheckpointResult] = []
    with tempfile.TemporaryDirectory(prefix=WORK_DIR_PREFIX) as work_dir:
        for name in models:
            results.extend(run_model(name, work_dir, cfg))
    return results
