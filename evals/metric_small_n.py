"""Small-n eval: when does metric learning beat no metric learning, and does AUTO pick right?

Four 12-d targets on x ~ U[0,1]^12 with noise 0.1 N(0,1):

- ``two_hi``: sin(6 pi x0) + sin(6 pi x1) (the ``metric_12d`` target; 2 of 12 inputs matter)
- ``two_lo``: sin(2 pi x0) + sin(2 pi x1) (same, lower frequency)
- ``sphere``: -4 |x - 0.5|^2 (all 12 inputs matter equally)
- ``lin3``: x0 + 2 x1 - x2 (3 of 12 inputs matter)

For each target and seed, rows stream to each n in ``n_grid`` through three BPANN_DISK
models: ``bpann_disk`` (NONE), ``bpann_disk_scale_x`` (divide inputs by their running standard
deviations) and ``bpann_disk_auto`` (Sobol/Var(x) weights, applied only if their leave-one-out
gain over the best isotropic metric is positive). ``gain`` is AUTO's latest gain (``na`` before
its first refit at 100 rows) and ``on`` whether AUTO used the learned metric.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from evals.flat_sphere import gaussian_loglik, rmse
from evals.metric_12d import NUM_DIM, Metric12dConfig, StreamedModel
from evals.stress_eval import format_larger, format_plain, format_smaller

NOISE_STD = 0.1
N_GRID: tuple[int, ...] = (10, 30, 100, 300)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MODELS: tuple[str, ...] = (
    "bpann_disk",
    "bpann_disk_scale_x",
    "bpann_disk_auto",
)
WORK_DIR_PREFIX = "enn_metric_small_n_"


def _two_hi(x: np.ndarray) -> np.ndarray:
    return np.sin(6 * np.pi * x[:, 0]) + np.sin(6 * np.pi * x[:, 1])


def _two_lo(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x[:, 0]) + np.sin(2 * np.pi * x[:, 1])


def _sphere(x: np.ndarray) -> np.ndarray:
    return -4.0 * ((x - 0.5) ** 2).sum(axis=1)


def _lin3(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + 2.0 * x[:, 1] - x[:, 2]


FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "two_hi": _two_hi,
    "two_lo": _two_lo,
    "sphere": _sphere,
    "lin3": _lin3,
}


@dataclass(frozen=True)
class SmallNConfig:
    functions: tuple[str, ...] = tuple(FUNCTIONS)
    seeds: tuple[int, ...] = SEEDS
    models: tuple[str, ...] = MODELS
    stream: Metric12dConfig = Metric12dConfig(n_grid=N_GRID)


@dataclass(frozen=True)
class SmallNResult:
    function: str
    seed: int
    model: str
    num_obs: int
    loglik: float
    nrmse: float
    gain: float | None
    on: bool | None


def make_data(
    function: str, num_obs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.random((num_obs, NUM_DIM))
    y = FUNCTIONS[function](x) + NOISE_STD * rng.standard_normal(num_obs)
    return x, y.reshape(-1, 1)


def _format_optional(value: float | None) -> str:
    return "na" if value is None else f"{value:.4f}"


def format_eval_line(r: SmallNResult) -> str:
    on = "na" if r.on is None else str(int(r.on))
    return (
        "EVAL: "
        f"{format_plain('function', r.function)} "
        f"{format_plain('seed', r.seed)} "
        f"{format_plain('model', r.model)} "
        f"{format_plain('n', r.num_obs)} "
        f"{format_larger('loglik', f'{r.loglik:.4f}')} "
        f"{format_smaller('nrmse', f'{r.nrmse:.4f}')} "
        f"{format_plain('gain', _format_optional(r.gain))} "
        f"{format_plain('on', on)}"
    )


def run_stream(
    function: str, seed: int, model: str, work_dir: str, config: SmallNConfig
) -> list[SmallNResult]:
    """Stream one target/seed through one model; one result per checkpoint."""
    stream = replace(config.stream, seed=seed)
    rng = np.random.default_rng(seed)
    x, y = make_data(function, max(stream.n_grid), rng)
    x_test, y_test = make_data(function, stream.num_test, rng)
    y_test_std = float(np.std(y_test))
    streamed = StreamedModel(model, work_dir, stream)
    results: list[SmallNResult] = []
    lo = 0
    for hi in stream.n_grid:
        streamed.advance(x, y, lo, hi)
        mu, se, _ = streamed.query(x_test)
        metric = None if streamed.model is None else streamed.model.metric
        gain = None if metric is None else metric.heldout_gain
        results.append(
            SmallNResult(
                function=function,
                seed=seed,
                model=model,
                num_obs=hi,
                loglik=gaussian_loglik(y_test, mu, se),
                nrmse=rmse(y_test, mu) / y_test_std,
                gain=gain,
                on=None if metric is None or gain is None else metric.uses_learned_metric,
            )
        )
        lo = hi
    return results


def summarize(results: list[SmallNResult]) -> list[str]:
    """Mean test loglik over seeds per (function, n, model), and AUTO's on-fraction."""
    lines = []
    keys = sorted({(r.function, r.num_obs) for r in results}, key=lambda t: (t[0], t[1]))
    for function, n in keys:
        cell = [r for r in results if r.function == function and r.num_obs == n]
        parts = []
        for model in dict.fromkeys(r.model for r in cell):
            rows = [r for r in cell if r.model == model]
            parts.append(f"{model}={np.mean([r.loglik for r in rows]):.3f}")
            ons = [r.on for r in rows if r.on is not None]
            if ons:
                parts.append(f"auto_on={sum(ons)}/{len(ons)}")
        lines.append(f"SUMMARY: function={function} n={n} mean_loglik " + " ".join(parts))
    return lines


def run_eval(config: SmallNConfig | None = None) -> list[SmallNResult]:
    cfg = SmallNConfig() if config is None else config
    s = cfg.stream
    print(
        f"num_dim={NUM_DIM} num_test={s.num_test} batch={s.batch} k={s.k} "
        f"functions={','.join(cfg.functions)} seeds={','.join(map(str, cfg.seeds))} "
        f"models={','.join(cfg.models)}",
        flush=True,
    )
    runs = [(f, s, m) for f in cfg.functions for s in cfg.seeds for m in cfg.models]
    results: list[SmallNResult] = []
    with tempfile.TemporaryDirectory(prefix=WORK_DIR_PREFIX) as work_dir:
        for function, seed, model in runs:
            rows = run_stream(function, seed, model, f"{work_dir}/{function}_{seed}", cfg)
            print("\n".join(format_eval_line(r) for r in rows), flush=True)
            results.extend(rows)
    for line in summarize(results):
        print(line, flush=True)
    return results
