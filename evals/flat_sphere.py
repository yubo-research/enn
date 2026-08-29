from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from evals.stress_eval import format_larger, format_plain, format_smaller
from ops.qa import gaussian_nll
from ops.stress import (
    DEFAULT_DRAW_K,
    DEFAULT_DRAW_NUM_FIT_CANDIDATES,
    DEFAULT_DRAW_NUM_FIT_SAMPLES,
    DEFAULT_DRAW_SEED,
    DEFAULT_NUM_DIM,
    DRAW_FLAGS,
    MeanSE,
    format_mean_se,
    make_draw_observations,
    mean_se,
)

NUM_DIM = DEFAULT_NUM_DIM
NUM_OBS = 10 * NUM_DIM
NUM_TEST = 100
NUM_SEEDS = 30


@dataclass(frozen=True)
class FlatSphereConfig:
    num_dim: int = NUM_DIM
    num_obs: int = NUM_OBS
    num_test: int = NUM_TEST
    seed: int = DEFAULT_DRAW_SEED
    num_seeds: int = NUM_SEEDS
    k: int = DEFAULT_DRAW_K
    num_fit_candidates: int = DEFAULT_DRAW_NUM_FIT_CANDIDATES
    num_fit_samples: int = DEFAULT_DRAW_NUM_FIT_SAMPLES
    index_driver: ENNIndexDriver = ENNIndexDriver.FLAT
    work_dir: str | None = None


@dataclass(frozen=True)
class FlatSphereSeedResult:
    loglik: float
    rmse: float


@dataclass(frozen=True)
class FlatSphereAggregate:
    num_dim: int
    num_obs: int
    num_test: int
    num_seeds: int
    seed: int
    loglik: MeanSE
    rmse: MeanSE


def gaussian_loglik(y: np.ndarray, mu: np.ndarray, se: np.ndarray) -> float:
    """Mean Gaussian predictive log-likelihood (larger is better)."""
    return float(-gaussian_nll(y, mu, se))


def rmse(y: np.ndarray, mu: np.ndarray) -> float:
    err = np.asarray(mu, dtype=float).ravel() - np.asarray(y, dtype=float).ravel()
    return float(np.sqrt(np.mean(err**2)))


def _build_sphere_enn(
    x: np.ndarray,
    y: np.ndarray,
    *,
    index_driver: ENNIndexDriver,
    work_dir: str | None,
    seed: int,
) -> EpistemicNearestNeighbors:
    model_kwargs: dict[str, object] = {
        "train_x": x,
        "train_y": y,
        "scale_x": False,
        "index_driver": index_driver,
    }
    if index_driver == ENNIndexDriver.BPANN_DISK:
        if work_dir is None:
            raise ValueError("bpann_disk requires work_dir")
        seed_dir = os.path.join(work_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        model_kwargs["work_dir"] = seed_dir
        model_kwargs["enn_storage"] = "disk"
    elif work_dir is not None:
        raise ValueError("work_dir requires bpann_disk")
    return EpistemicNearestNeighbors(**model_kwargs)


def run_flat_sphere_seed(config: FlatSphereConfig) -> FlatSphereSeedResult:
    """Fit ENN on sphere DGP; score loglik and rmse on a held-out test set."""
    if config.num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if config.num_test < 1:
        raise ValueError("num_test must be >= 1")
    if config.num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    if config.k < 1:
        raise ValueError("k must be >= 1")

    data_rng = np.random.default_rng(config.seed)
    fit_rng = np.random.default_rng(config.seed + 1)
    x, y = make_draw_observations(
        config.num_obs, num_dim=config.num_dim, rng=data_rng
    )
    x_test, y_test = make_draw_observations(
        config.num_test, num_dim=config.num_dim, rng=data_rng
    )
    model = _build_sphere_enn(
        x,
        y,
        index_driver=config.index_driver,
        work_dir=config.work_dir,
        seed=config.seed,
    )
    fitted = enn_fit(
        model,
        k=config.k,
        num_fit_candidates=config.num_fit_candidates,
        num_fit_samples=config.num_fit_samples,
        rng=fit_rng,
    )
    post = model.posterior(x_test, params=fitted, flags=DRAW_FLAGS)
    return FlatSphereSeedResult(
        loglik=gaussian_loglik(y_test, post.mu, post.se),
        rmse=rmse(y_test, post.mu),
    )


def run_flat_sphere_over_seeds(config: FlatSphereConfig) -> FlatSphereAggregate:
    """Run sphere scoring for ``seed .. seed+num_seeds-1`` and aggregate."""
    if config.num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    results = [
        run_flat_sphere_seed(
            FlatSphereConfig(
                num_dim=config.num_dim,
                num_obs=config.num_obs,
                num_test=config.num_test,
                seed=config.seed + i,
                num_seeds=1,
                k=config.k,
                num_fit_candidates=config.num_fit_candidates,
                num_fit_samples=config.num_fit_samples,
                index_driver=config.index_driver,
                work_dir=config.work_dir,
            )
        )
        for i in range(config.num_seeds)
    ]
    return FlatSphereAggregate(
        num_dim=config.num_dim,
        num_obs=config.num_obs,
        num_test=config.num_test,
        num_seeds=config.num_seeds,
        seed=config.seed,
        loglik=mean_se([r.loglik for r in results]),
        rmse=mean_se([r.rmse for r in results]),
    )


def format_flat_sphere_eval_line(result: FlatSphereAggregate) -> str:
    return (
        "EVAL: "
        f"{format_plain('n', result.num_obs)} "
        f"{format_larger('loglik', format_mean_se(result.loglik))} "
        f"{format_smaller('rmse', format_mean_se(result.rmse))}"
    )


def run_flat_sphere_eval(config: FlatSphereConfig | None = None) -> FlatSphereAggregate:
    """Aggregate sphere metrics and print the EVAL line."""
    cfg = FlatSphereConfig() if config is None else config
    result = run_flat_sphere_over_seeds(cfg)
    print(
        f"num_dim={result.num_dim} num_obs={result.num_obs} "
        f"num_test={result.num_test} seed={result.seed} "
        f"num_seeds={result.num_seeds} index_driver={cfg.index_driver.name}",
        flush=True,
    )
    print(format_flat_sphere_eval_line(result), flush=True)
    return result
