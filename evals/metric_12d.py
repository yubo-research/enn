"""Streaming eval on the 12-d "two of twelve" problem, where metric learning matters.

x ~ U[0,1]^12, y = sin(6 pi x0) + sin(6 pi x1) + 0.1 eps, so ten inputs are irrelevant
(reports/mbpann/metric_advantage.py). Rows stream in batches with ``ensure_index_sync``
after each batch. At every checkpoint each model refits its ENN hyperparameters with
``enn_fit``; ``bpann_disk_metric_learning`` first refits diagonal metric weights by LOOCV
(reports/iaml/iaml_core.fit_exact) on a subsample and applies them with
``MBPANNMetric.set_weights``. Once the previous checkpoint already filled the fit
subsample, the metric fit is warm-started from the previous checkpoint's metric
(``metric_warm_start``); earlier fits start cold, since a warm start from a fit on a
few dozen rows stays stuck at the identity-metric level. ``bpann_disk_auto`` applies the
learned metric only when ``heldout_metric_gain`` validates it
(``ENNMetricLearning.AUTO``); see ``evals/metric_small_n.py``.

``add_s`` is wall time from the previous checkpoint to this one: adds, syncs, metric
fit, and hyperparameter fit. ``query_s`` is one posterior call on the test set.

The whole stream is repeated for seeds ``seed .. seed+num_seeds-1`` (data drawn from
``seed``, fits from ``seed + 1``); each EVAL line reports mean ± standard error over seeds.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import ModuleType

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.mbpann import MBPANNMetric
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_x_scaling import ENNMetricLearning, ENNScaleX
from evals.flat_sphere import gaussian_loglik, rmse
from evals.metric_12d_results import (
    CheckpointResult,
    CheckpointSummary,
    format_eval_line,
    format_seed_line,
    summarize,
)
from evals.stress_eval import REPO_ROOT
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
NUM_SEEDS = 10
MODELS: tuple[str, ...] = (
    "flat",
    "flat_scale_x",
    "bpann_disk",
    "bpann_disk_metric_learning",
    "bpann_disk_auto",
)
BPANN_METRIC_LEARNING: dict[str, ENNMetricLearning] = {
    "bpann_disk": ENNMetricLearning.OFF,
    "bpann_disk_metric_learning": ENNMetricLearning.ON,
    "bpann_disk_auto": ENNMetricLearning.AUTO,
}
LEARNED_METRIC_MODELS = frozenset(
    name for name, mode in BPANN_METRIC_LEARNING.items() if mode != ENNMetricLearning.OFF
)
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
    num_seeds: int = NUM_SEEDS


@functools.cache
def load_iaml_core() -> ModuleType:
    spec = importlib.util.spec_from_file_location("iaml_core", IAML_CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {IAML_CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DataFn = Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]]


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
        return EpistemicNearestNeighbors(x, y, scale_x=ENNScaleX.ON)
    if name not in BPANN_METRIC_LEARNING:
        raise ValueError(f"unknown model {name!r}")
    model_dir = os.path.join(work_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    return EpistemicNearestNeighbors(
        x,
        y,
        metric_learning=BPANN_METRIC_LEARNING[name],
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=model_dir,
    )


@dataclass(frozen=True)
class MetricFit:
    """``theta`` of the applied learned metric (None if AUTO kept the identity metric) and, in
    AUTO mode, the held-out gain that decided it."""

    theta: np.ndarray | None
    heldout_gain: float | None = None


def _heldout_ll_diff(
    core: ModuleType, x_fit: np.ndarray, y_fit: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, k: int
) -> np.ndarray:
    k_fit = min(k, len(x_fit) - 1)
    full = core.Metric(x_fit.shape[1])
    core.fit_exact(full, x_fit, y_fit, k_fit)
    iso = core.Metric(x_fit.shape[1], isotropic=True)
    core.fit_exact(iso, x_fit, y_fit, k_fit)
    k_val = min(k, len(x_fit))
    ll_full, ll_iso = (
        core.loglik_grad(x_val, y_val, x_fit, y_fit, core.knn(x_val, x_fit, k_val, m.a), m.a, m.c)[0]
        for m in (full, iso)
    )
    return ll_full - ll_iso


def heldout_metric_gain(x: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator) -> float:
    """2-fold held-out gain of a LOOCV-fit diagonal metric over the best isotropic metric.

    Rows are split in half; both metrics are fit on one half and scored (mean Gaussian
    log-likelihood, neighbors from the fitting half) on the other, then the halves swap.
    Returns the mean per-row difference, or ``-inf`` with fewer than 4 rows.
    """
    if len(x) < 4:
        return float("-inf")
    core = load_iaml_core()
    perm = rng.permutation(len(x))
    halves = (perm[: len(x) // 2], perm[len(x) // 2 :])
    diffs = [
        _heldout_ll_diff(core, x[fit], y[fit], x[val], y[val], k)
        for fit, val in (halves, halves[::-1])
    ]
    return float(np.concatenate(diffs).mean())


def fit_metric(
    helper: MBPANNMetric,
    x: np.ndarray,
    y: np.ndarray,
    config: Metric12dConfig,
    rng: np.random.Generator,
    prev_theta: np.ndarray | None = None,
) -> MetricFit:
    """LOOCV-fit diagonal metric weights on a subsample and apply them to the index.

    With ``prev_theta`` the fit is warm-started there: one outer round, no isotropic restart.
    In AUTO mode the subsample's held-out gain is computed first; if it rejects metric
    learning, the identity metric is applied and the full fit is skipped.
    """
    iaml_core = load_iaml_core()
    num_obs = len(x)
    sub = rng.choice(num_obs, size=min(num_obs, config.metric_fit_subsample), replace=False)
    x_sub, y_sub = x[sub], y[sub, 0]
    gain = None
    if helper.metric_learning == ENNMetricLearning.AUTO:
        gain = heldout_metric_gain(x_sub, y_sub, config.k, rng)
        if not helper.set_weights_if_validated(helper.weights, gain):
            return MetricFit(theta=None, heldout_gain=gain)
    metric = iaml_core.Metric(x.shape[1])
    k = min(config.k, len(sub) - 1)
    if prev_theta is None:
        iaml_core.fit_exact(metric, x_sub, y_sub, k)
    else:
        metric.theta = prev_theta.copy()
        iaml_core.fit_exact(metric, x_sub, y_sub, k, outer=1, restart=False)
    if gain is None:
        helper.set_weights(metric.a)
    else:
        helper.set_weights_if_validated(metric.a, gain)
    return MetricFit(theta=metric.theta.copy(), heldout_gain=gain)


class StreamedModel:
    """One model grown checkpoint to checkpoint on a shared row stream."""

    def __init__(self, name: str, work_dir: str, config: Metric12dConfig) -> None:
        self.name = name
        self.work_dir = work_dir
        self.config = config
        self.model: EpistemicNearestNeighbors | None = None
        self.helper: MBPANNMetric | None = None
        self.params: object = None
        self.metric_fit: MetricFit | None = None
        self.fit_rng = np.random.default_rng(config.seed + 1)
        if name in LEARNED_METRIC_MODELS:
            load_iaml_core()

    def _add_rows(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.model is None:
            self.model = build_model(self.name, x, y, self.work_dir)
            if self.model.metric_learning != ENNMetricLearning.OFF:
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
            prev = self.metric_fit.theta if warm and self.metric_fit is not None else None
            self.metric_fit = fit_metric(
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


def run_model(
    name: str, work_dir: str, config: Metric12dConfig, data: DataFn = make_data
) -> list[CheckpointResult]:
    """Stream ``config.seed``'s data through one model; print and return one row per checkpoint."""
    rng = np.random.default_rng(config.seed)
    x, y = data(max(config.n_grid), rng)
    x_test, y_test = data(config.num_test, rng)
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
        print(format_seed_line(config.seed, result), flush=True)
        results.append(result)
        lo = hi
    return results


def run_eval(
    config: Metric12dConfig | None = None,
    models: tuple[str, ...] = MODELS,
    data: DataFn = make_data,
) -> list[CheckpointSummary]:
    cfg = Metric12dConfig() if config is None else config
    if list(cfg.n_grid) != sorted(set(cfg.n_grid)) or cfg.n_grid[0] < 2:
        raise ValueError("n_grid must be strictly increasing and start at >= 2")
    if cfg.num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    print(
        f"num_dim={NUM_DIM} num_test={cfg.num_test} batch={cfg.batch} k={cfg.k} "
        f"seed={cfg.seed} num_seeds={cfg.num_seeds} models={','.join(models)}",
        flush=True,
    )
    results: list[CheckpointResult] = []
    for i in range(cfg.num_seeds):
        seed_cfg = replace(cfg, seed=cfg.seed + i)
        with tempfile.TemporaryDirectory(prefix=WORK_DIR_PREFIX) as work_dir:
            for name in models:
                results.extend(run_model(name, work_dir, seed_cfg, data))
    summaries = summarize(results)
    for summary in summaries:
        print(format_eval_line(summary), flush=True)
    return summaries
