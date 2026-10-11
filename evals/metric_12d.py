
from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNStorage
from enn.turbo.config.enn_x_scaling import ENNMetricLearning, ENNScaleX
from evals.flat_sphere import gaussian_loglik, rmse
from evals.metric_12d_results import (
    CheckpointResult,
    CheckpointSummary,
    format_eval_line,
    format_seed_line,
    summarize,
)
from ops.stress import DRAW_FLAGS

NUM_DIM = 12
NOISE_STD = 0.1
N_GRID: tuple[int, ...] = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000)
NUM_TEST = 1000
BATCH = 500
K = 10
NUM_FIT_CANDIDATES = 100
NUM_FIT_SAMPLES = 100
SEED = 0
NUM_SEEDS = 10
MODELS: tuple[str, ...] = (
    "flat",
    "flat_scale_x",
    "bpann_disk",
    "bpann_disk_auto",
)
BPANN_METRIC_LEARNING: dict[str, ENNMetricLearning] = {
    "bpann_disk": ENNMetricLearning.NONE,
    "bpann_disk_scale_x": ENNMetricLearning.NONE,
    "bpann_disk_auto": ENNMetricLearning.AUTO,
}
WORK_DIR_PREFIX = "enn_metric_12d_"


@dataclass(frozen=True)
class Metric12dConfig:
    n_grid: tuple[int, ...] = N_GRID
    num_test: int = NUM_TEST
    batch: int = BATCH
    k: int = K
    num_fit_candidates: int = NUM_FIT_CANDIDATES
    num_fit_samples: int = NUM_FIT_SAMPLES
    seed: int = SEED
    num_seeds: int = NUM_SEEDS
    num_rows: int | None = None

    def stream_rows(self) -> int:
        return max(self.n_grid) if self.num_rows is None else self.num_rows


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
        scale_x=ENNScaleX.ON if name == "bpann_disk_scale_x" else ENNScaleX.OFF,
        metric_learning=BPANN_METRIC_LEARNING[name],
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=model_dir,
        enn_storage=ENNStorage.DISK,
    )


class StreamedModel:

    def __init__(self, name: str, work_dir: str, config: Metric12dConfig) -> None:
        self.name = name
        self.work_dir = work_dir
        self.config = config
        self.model: EpistemicNearestNeighbors | None = None
        self.params: object = None
        self.fit_rng = np.random.default_rng(config.seed + 1)

    def _add_rows(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.model is None:
            self.model = build_model(self.name, x, y, self.work_dir)
        else:
            self.model.add(x, y)
        self.model.ensure_index_sync()

    def advance(self, x: np.ndarray, y: np.ndarray, lo: int, hi: int) -> float:
        t0 = time.perf_counter()
        for start in range(lo, hi, self.config.batch):
            stop = min(hi, start + self.config.batch)
            self._add_rows(x[start:stop], y[start:stop])
        assert self.model is not None
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
    rng = np.random.default_rng(config.seed)
    x, y = data(config.stream_rows(), rng)
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
    if cfg.stream_rows() < max(cfg.n_grid):
        raise ValueError("num_rows must be >= max(n_grid)")
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
