"""Streaming evals with categorical inputs, one-hot encoded: categorical-only and mixed.

Each categorical variable with ``c`` categories is ``c`` one-hot columns in {0, 1} (a variable with
one category is a constant column). Level effects are fixed tables drawn once from ``EFFECT_SEED``,
centered and scaled to unit variance, then multiplied by the variable's strength. Noise is 0.1 N(0,1).

- ``cat_only``: six categorical variables with 1, 2, 3, 4, 6 and 10 categories (26 columns) and
  main-effect strengths 0, 3, 1, 0.3, 0.1 and 0 (the 10-category variable is irrelevant), plus an
  interaction of strength 1 between the 3- and 4-category variables. The interaction table is
  centered only overall, so part of it acts as main effects; most of it (about two thirds of its
  variance) is pure interaction, which first-order Sobol indices do not see, and the 4-category
  variable matters mostly through that part.
- ``mixed``: four continuous inputs x_i = s_i u_i, u ~ U[0,1]^4, ranges s = 0.01, 1, 100, 10, and
  three categorical variables with 3, 5 and 2 categories (14 columns). The target is
  sin(2 pi u0) (strong, narrowest input) + slope[A] (u1 - 0.5) (the slope on u1 depends on the
  3-category variable A) + 0.3 (u2 - 0.5) (weak, widest input) + A's main effect (strength 1) +
  the 5-category variable's (strength 0.3); u3 and the 2-category variable are irrelevant.

Models are BPANN_DISK: ``bpann_disk`` (raw distances), ``bpann_disk_scale_x`` (every column divided
by its running standard deviation), ``bpann_disk_scale_x_tied`` (the same, but ``tied_dims`` leaves
the one-hot columns unscaled), ``bpann_disk_auto`` (metric_learning AUTO, one weight per column) and
``bpann_disk_auto_tied`` (AUTO with each variable's one-hot columns tied to one joint-Sobol weight).
The streaming protocol, seeds and mean ± SE lines are those of ``evals/metric_12d.py``; each EVAL
line also names the problem.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, replace

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNStorage
from enn.turbo.config.enn_x_scaling import ENNMetricLearning, ENNScaleX
from evals.flat_sphere import gaussian_loglik, rmse
from evals.metric_12d import DataFn, Metric12dConfig, StreamedModel
from evals.metric_12d_results import (
    CheckpointResult,
    CheckpointSummary,
    format_eval_line,
    format_seed_line,
    summarize,
)
from evals.stress_eval import format_plain

NOISE_STD = 0.1
EFFECT_SEED = 20260929
N_GRID: tuple[int, ...] = (30, 100, 300, 1000, 3000)
NUM_SEEDS = 5
CONFIG = Metric12dConfig(n_grid=N_GRID, num_test=500, num_seeds=NUM_SEEDS)
MODELS: tuple[str, ...] = (
    "bpann_disk",
    "bpann_disk_scale_x",
    "bpann_disk_scale_x_tied",
    "bpann_disk_auto",
    "bpann_disk_auto_tied",
)
WORK_DIR_PREFIX = "enn_metric_categorical_"

CAT_ONLY_CARDS: tuple[int, ...] = (1, 2, 3, 4, 6, 10)
CAT_ONLY_STRENGTHS: tuple[float, ...] = (0.0, 3.0, 1.0, 0.3, 0.1, 0.0)
CAT_ONLY_INTERACTION = (2, 3, 1.0)
MIXED_RANGES: tuple[float, ...] = (0.01, 1.0, 100.0, 10.0)
MIXED_CARDS: tuple[int, ...] = (3, 5, 2)
MIXED_STRENGTHS: tuple[float, ...] = (1.0, 0.3, 0.0)
MIXED_SLOPES = np.array([1.0, -1.0, 2.0])


def _unit_table(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    t = rng.standard_normal(shape)
    t = t - t.mean()
    sd = t.std()
    return t / sd if sd > 0 else t


def _tables() -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(EFFECT_SEED)
    cat_only = [s * _unit_table((c,), rng) for c, s in zip(CAT_ONLY_CARDS, CAT_ONLY_STRENGTHS)]
    a, b, s = CAT_ONLY_INTERACTION
    pair = s * _unit_table((CAT_ONLY_CARDS[a], CAT_ONLY_CARDS[b]), rng)
    mixed = [s * _unit_table((c,), rng) for c, s in zip(MIXED_CARDS, MIXED_STRENGTHS)]
    return cat_only, pair, mixed


CAT_ONLY_EFFECTS, CAT_ONLY_PAIR, MIXED_EFFECTS = _tables()


def tied_groups(cards: tuple[int, ...], start: int = 0) -> tuple[tuple[int, ...], ...]:
    """Column indices of each variable's one-hot block, blocks laid out from column ``start``."""
    ends = start + np.cumsum(cards)
    return tuple(tuple(range(e - c, e)) for c, e in zip(cards, ends.tolist()))


def _categories(
    cards: tuple[int, ...], num_obs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform category of each variable (``num_obs`` x vars) and its one-hot columns."""
    levels = np.column_stack([rng.integers(0, c, num_obs) for c in cards])
    onehot = np.hstack([np.eye(c)[levels[:, i]] for i, c in enumerate(cards)])
    return levels, onehot


def _noisy(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (f + NOISE_STD * rng.standard_normal(len(f))).reshape(-1, 1)


def make_cat_only(num_obs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    levels, x = _categories(CAT_ONLY_CARDS, num_obs, rng)
    f = sum(t[levels[:, i]] for i, t in enumerate(CAT_ONLY_EFFECTS))
    a, b, _ = CAT_ONLY_INTERACTION
    f = f + CAT_ONLY_PAIR[levels[:, a], levels[:, b]]
    return x, _noisy(f, rng)


def make_mixed(num_obs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.random((num_obs, len(MIXED_RANGES)))
    levels, onehot = _categories(MIXED_CARDS, num_obs, rng)
    f = (
        np.sin(2 * np.pi * u[:, 0])
        + MIXED_SLOPES[levels[:, 0]] * (u[:, 1] - 0.5)
        + 0.3 * (u[:, 2] - 0.5)
        + sum(t[levels[:, i]] for i, t in enumerate(MIXED_EFFECTS))
    )
    return np.hstack([u * np.array(MIXED_RANGES), onehot]), _noisy(f, rng)


@dataclass(frozen=True)
class Problem:
    name: str
    data: DataFn
    tied_dims: tuple[tuple[int, ...], ...]


PROBLEMS: tuple[Problem, ...] = (
    Problem("cat_only", make_cat_only, tied_groups(CAT_ONLY_CARDS)),
    Problem("mixed", make_mixed, tied_groups(MIXED_CARDS, start=len(MIXED_RANGES))),
)


def build_model(
    name: str, x: np.ndarray, y: np.ndarray, work_dir: str, tied_dims: tuple[tuple[int, ...], ...]
) -> EpistemicNearestNeighbors:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}")
    model_dir = os.path.join(work_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    auto = name.startswith("bpann_disk_auto")
    return EpistemicNearestNeighbors(
        x,
        y,
        tied_dims=tied_dims if name.endswith("_tied") else None,
        scale_x=ENNScaleX.ON if "scale_x" in name else ENNScaleX.OFF,
        metric_learning=ENNMetricLearning.AUTO if auto else ENNMetricLearning.NONE,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=model_dir,
        enn_storage=ENNStorage.DISK,
    )


class CategoricalStreamedModel(StreamedModel):
    def __init__(
        self, name: str, work_dir: str, config: Metric12dConfig, tied_dims: tuple[tuple[int, ...], ...]
    ) -> None:
        super().__init__(name, work_dir, config)
        self.tied_dims = tied_dims

    def _add_rows(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.model is None:
            self.model = build_model(self.name, x, y, self.work_dir, self.tied_dims)
        else:
            self.model.add(x, y)
        self.model.ensure_index_sync()


def run_model(
    problem: Problem, name: str, work_dir: str, config: Metric12dConfig
) -> list[CheckpointResult]:
    """Stream ``config.seed``'s rows of ``problem`` through one model; one result per checkpoint."""
    rng = np.random.default_rng(config.seed)
    x, y = problem.data(config.stream_rows(), rng)
    x_test, y_test = problem.data(config.num_test, rng)
    y_test_std = float(np.std(y_test))
    streamed = CategoricalStreamedModel(name, work_dir, config, problem.tied_dims)
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
        print(f"{format_plain('problem', problem.name)} {format_seed_line(config.seed, result)}", flush=True)
        results.append(result)
        lo = hi
    return results


def format_problem_eval_line(problem: str, summary: CheckpointSummary) -> str:
    line = format_eval_line(summary)
    return f"EVAL: {format_plain('problem', problem)} {line.removeprefix('EVAL: ')}"


def run_problem(problem: Problem, config: Metric12dConfig, models: tuple[str, ...]) -> list[CheckpointSummary]:
    results: list[CheckpointResult] = []
    for i in range(config.num_seeds):
        seed_cfg = replace(config, seed=config.seed + i)
        with tempfile.TemporaryDirectory(prefix=WORK_DIR_PREFIX) as work_dir:
            for name in models:
                results.extend(run_model(problem, name, work_dir, seed_cfg))
    return summarize(results)


def run_eval(
    config: Metric12dConfig = CONFIG,
    models: tuple[str, ...] = MODELS,
    problems: tuple[Problem, ...] = PROBLEMS,
) -> dict[str, list[CheckpointSummary]]:
    print(
        f"num_test={config.num_test} batch={config.batch} k={config.k} seed={config.seed} "
        f"num_seeds={config.num_seeds} models={','.join(models)} "
        f"problems={','.join(p.name for p in problems)}",
        flush=True,
    )
    out: dict[str, list[CheckpointSummary]] = {}
    for problem in problems:
        out[problem.name] = run_problem(problem, config, models)
        for summary in out[problem.name]:
            print(format_problem_eval_line(problem.name, summary), flush=True)
    return out
