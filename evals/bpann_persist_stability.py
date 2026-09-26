from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.enn_params import PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from evals.stress_eval import format_plain, format_smaller

NUM_DIM = 10
NUM_METRICS = 3
K = 10
NUM_FIT_CANDIDATES = 100
NUM_FIT_SAMPLES = 100
NUM_LOO = 100
NUM_TEST = 100
N_GRID = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)
SEED = 0
NOISE_STD = 0.1
DRAW_F_CENTER = 0.3
WORK_DIR_PREFIX = "enn_bpann_persist_stability_"
FLAGS_LOO = PosteriorFlags(exclude_nearest=True, observation_noise=True)
FLAGS_TEST = PosteriorFlags(exclude_nearest=False, observation_noise=True)


@dataclass(frozen=True)
class PersistConfig:
    num_obs: int
    work_dir: str
    seed: int = SEED
    k: int = K
    num_fit_candidates: int = NUM_FIT_CANDIDATES
    num_fit_samples: int = NUM_FIT_SAMPLES
    num_loo: int = NUM_LOO
    num_test: int = NUM_TEST


def rmse(y: np.ndarray, mu: np.ndarray) -> float:
    err = np.asarray(mu, dtype=float).ravel() - np.asarray(y, dtype=float).ravel()
    return float(np.sqrt(np.mean(err**2)))


def make_synthetic(
    num_obs: int,
    *,
    num_dim: int,
    num_metrics: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    if num_metrics < 1:
        raise ValueError("num_metrics must be >= 1")
    x = rng.uniform(0.0, 1.0, size=(num_obs, num_dim))
    base = np.sum((x - DRAW_F_CENTER) ** 2, axis=1, keepdims=True)
    scales = np.linspace(0.5, 1.5, num_metrics).reshape(1, -1)
    noise = rng.standard_normal((num_obs, num_metrics))
    y = base * scales + NOISE_STD * noise
    return x, y


def build_model(
    x: np.ndarray, y: np.ndarray, work_dir: str
) -> EpistemicNearestNeighbors:
    return EpistemicNearestNeighbors(
        x,
        y,
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=work_dir,
        enn_storage="disk",
    )


def loocv_rmse(
    model: EpistemicNearestNeighbors,
    x: np.ndarray,
    y: np.ndarray,
    params: object,
) -> float:
    post = model.posterior(x, params=params, flags=FLAGS_LOO)
    return rmse(y, post.mu)


def holdout_rmse(
    model: EpistemicNearestNeighbors,
    x: np.ndarray,
    y: np.ndarray,
    params: object,
) -> float:
    post = model.posterior(x, params=params, flags=FLAGS_TEST)
    return rmse(y, post.mu)


def _validate_config(config: PersistConfig) -> None:
    if config.num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if config.k < 1:
        raise ValueError("k must be >= 1")
    if config.num_loo < 1:
        raise ValueError("num_loo must be >= 1")
    if config.num_test < 1:
        raise ValueError("num_test must be >= 1")


@dataclass(frozen=True)
class ScorePair:
    loocv: float
    rmse: float


def _score_pair(
    model: EpistemicNearestNeighbors,
    x_o0: np.ndarray,
    y_o0: np.ndarray,
    x_o1: np.ndarray,
    y_o1: np.ndarray,
    params: object,
) -> ScorePair:
    return ScorePair(
        loocv_rmse(model, x_o0, y_o0, params),
        holdout_rmse(model, x_o1, y_o1, params),
    )


def _abs_score_diffs(before: ScorePair, after: ScorePair) -> tuple[float, float]:
    return abs(before.loocv - after.loocv), abs(before.rmse - after.rmse)


def _loaded_score_pair(
    work_dir: str,
    x_o0: np.ndarray,
    y_o0: np.ndarray,
    x_o1: np.ndarray,
    y_o1: np.ndarray,
    params: object,
) -> ScorePair:
    loaded = build_model(
        np.empty((0, NUM_DIM)), np.empty((0, NUM_METRICS)), work_dir
    )
    loaded.ensure_index_sync()
    scores = _score_pair(loaded, x_o0, y_o0, x_o1, y_o1, params)
    del loaded
    return scores


@dataclass(frozen=True)
class InnerData:
    x_n: np.ndarray
    y_n: np.ndarray
    x_o0: np.ndarray
    y_o0: np.ndarray
    x_o1: np.ndarray
    y_o1: np.ndarray
    n_dir: str


def _prepare_inner_data(config: PersistConfig) -> InnerData:
    rng = np.random.default_rng(config.seed)
    x_n, y_n = make_synthetic(
        config.num_obs, num_dim=NUM_DIM, num_metrics=NUM_METRICS, rng=rng
    )
    n_loo = min(config.num_loo, config.num_obs)
    x_o1, y_o1 = make_synthetic(
        config.num_test, num_dim=NUM_DIM, num_metrics=NUM_METRICS, rng=rng
    )
    n_dir = os.path.join(config.work_dir, f"n_{config.num_obs}")
    os.makedirs(n_dir, exist_ok=True)
    return InnerData(
        x_n=x_n,
        y_n=y_n,
        x_o0=x_n[:n_loo],
        y_o0=y_n[:n_loo],
        x_o1=x_o1,
        y_o1=y_o1,
        n_dir=n_dir,
    )


def run_inner(config: PersistConfig) -> tuple[float, float, float, float]:
    _validate_config(config)
    data = _prepare_inner_data(config)
    fit_rng = np.random.default_rng(config.seed + 1)
    model = build_model(
        np.empty((0, NUM_DIM)), np.empty((0, NUM_METRICS)), data.n_dir
    )
    model.add(data.x_n, data.y_n)
    params = enn_fit(
        model,
        k=config.k,
        num_fit_candidates=config.num_fit_candidates,
        num_fit_samples=config.num_fit_samples,
        rng=fit_rng,
    )
    before = _score_pair(
        model, data.x_o0, data.y_o0, data.x_o1, data.y_o1, params
    )
    model.persist_index_to_disk()
    loaded = _loaded_score_pair(
        data.n_dir, data.x_o0, data.y_o0, data.x_o1, data.y_o1, params
    )
    after = _score_pair(
        model, data.x_o0, data.y_o0, data.x_o1, data.y_o1, params
    )
    d_loo, d_rmse = _abs_score_diffs(before, after)
    d_loo_load, d_rmse_load = _abs_score_diffs(before, loaded)
    return d_loo, d_rmse, d_loo_load, d_rmse_load



def format_eval_line(
    num_obs: int,
    abs_loocv: float,
    abs_rmse: float,
    abs_loocv_load: float,
    abs_rmse_load: float,
) -> str:
    return (
        "EVAL: "
        f"{format_plain('n', num_obs)} "
        f"{format_smaller('abs_loocv_diff', abs_loocv)} "
        f"{format_smaller('abs_rmse_diff', abs_rmse)} "
        f"{format_smaller('abs_loocv_load_diff', abs_loocv_load)} "
        f"{format_smaller('abs_rmse_load_diff', abs_rmse_load)}"
    )


def run_grid(
    n_grid: tuple[int, ...] = N_GRID,
    *,
    work_dir: str,
    seed: int = SEED,
    k: int = K,
    num_fit_candidates: int = NUM_FIT_CANDIDATES,
    num_fit_samples: int = NUM_FIT_SAMPLES,
) -> list[tuple[int, float, float, float, float]]:
    out: list[tuple[int, float, float, float, float]] = []
    for n in n_grid:
        d_loo, d_rmse, d_loo_load, d_rmse_load = run_inner(
            PersistConfig(
                num_obs=n,
                work_dir=work_dir,
                seed=seed,
                k=k,
                num_fit_candidates=num_fit_candidates,
                num_fit_samples=num_fit_samples,
            )
        )
        print(
            format_eval_line(n, d_loo, d_rmse, d_loo_load, d_rmse_load),
            flush=True,
        )
        out.append((n, d_loo, d_rmse, d_loo_load, d_rmse_load))
    return out


def run_eval(
    n_grid: tuple[int, ...] = N_GRID,
    *,
    work_dir: str | None = None,
    seed: int = SEED,
    k: int = K,
    num_fit_candidates: int = NUM_FIT_CANDIDATES,
    num_fit_samples: int = NUM_FIT_SAMPLES,
) -> list[tuple[int, float, float, float, float]]:
    if work_dir is not None:
        return run_grid(
            n_grid,
            work_dir=work_dir,
            seed=seed,
            k=k,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
        )
    with tempfile.TemporaryDirectory(prefix=WORK_DIR_PREFIX) as wd:
        return run_grid(
            n_grid,
            work_dir=wd,
            seed=seed,
            k=k,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
        )
