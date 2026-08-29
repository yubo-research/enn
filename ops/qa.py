#!/usr/bin/env python

from __future__ import annotations

import tempfile
import time

import click
import numpy as np

from dataclasses import dataclass

from enn import create_optimizer, turbo_enn_config
from enn.benchmarks import Ackley
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.enn_params import ENNParams
from enn.turbo.config import (
    AcqType,
    CandidateGenConfig,
    ENNFitConfig,
    ENNIndexDriver,
    ENNSurrogateConfig,
    TurboTRConfig,
)
from enn.turbo.optimizer_config import CandidateRV

NUM_DIM = 99
NUM_ROUNDS = 101
NUM_ARMS = 100
ACKLEY_NOISE = 0.1
SEED = 0
INDEX_TYPE_CHOICES: tuple[str, ...] = ("flat", "bpann_disk")
Y_BOUNDS_N_TRAIN = 80
Y_BOUNDS_N_TEST = 200
Y_BOUNDS_K = 10
Y_BOUNDS_NUM_FIT_CANDIDATES = 30
Y_BOUNDS_NUM_FIT_SAMPLES = 20
Y_BOUNDS_NUM_DRAWS = 64
Y_BOUNDS_LOG_NOISE = 0.15


def parse_index_driver(name: str) -> ENNIndexDriver:
    mapping = {
        "flat": ENNIndexDriver.FLAT,
        "bpann_disk": ENNIndexDriver.BPANN_DISK,
    }
    if name not in mapping:
        raise ValueError(f"Unknown index type: {name}")
    return mapping[name]


def build_ackley_enn_surrogate(index_driver: ENNIndexDriver) -> ENNSurrogateConfig:
    """Build ENN surrogate config; BPANN_DISK requires disk storage + work_dir."""
    enn_kwargs: dict[str, object] = {
        "k": 10,
        "fit": ENNFitConfig(num_fit_samples=100),
        "index_driver": index_driver,
    }
    if index_driver == ENNIndexDriver.BPANN_DISK:
        enn_kwargs["enn_storage"] = "disk"
        enn_kwargs["work_dir"] = tempfile.mkdtemp(prefix="qa_ackley_bpann_")
    return ENNSurrogateConfig(**enn_kwargs)


def run_turbo_enn_ackley(
    *,
    index_driver: ENNIndexDriver,
    num_dim: int = NUM_DIM,
    num_rounds: int = NUM_ROUNDS,
    num_arms: int = NUM_ARMS,
    noise: float = ACKLEY_NOISE,
    seed: int = SEED,
) -> None:
    """Run TuRBO-ENN on Ackley; print EVAL metrics each round."""
    rng = np.random.default_rng(seed)
    objective = Ackley(noise=noise, rng=rng)
    bounds = np.array([objective.bounds] * num_dim, dtype=float)
    config = turbo_enn_config(
        enn=build_ackley_enn_surrogate(index_driver),
        candidates=CandidateGenConfig(candidate_rv=CandidateRV.UNIFORM),
        trust_region=TurboTRConfig(noise_aware=True),
        acq_type=AcqType.UCB,
    )
    optimizer = create_optimizer(
        bounds=bounds,
        config=config,
        rng=np.random.default_rng(seed),
    )
    y_var_scale = float(noise) ** 2
    y_best = -np.inf
    cumulative_num_arms = 0
    t_start = time.perf_counter()
    for i_iter in range(num_rounds):
        x_arms = optimizer.ask(num_arms=num_arms)
        y_obs = np.asarray(objective(x_arms), dtype=float).reshape(-1, 1)
        optimizer.tell(x_arms, y_obs, y_var=y_var_scale * np.ones_like(y_obs))
        y_best = max(y_best, float(np.max(y_obs)))
        cumulative_num_arms += int(np.asarray(x_arms).shape[0])
        seconds_since_start_of_run = time.perf_counter() - t_start
        click.echo(
            f"EVAL: iter = {i_iter} arms = {cumulative_num_arms} "
            f"SMALLER(dt) = {seconds_since_start_of_run:.02f} "
            f"LARGER(y_best) = {y_best:.04f}"
        )


@click.group()
def cli() -> None:
    """QA / optimization-performance checks."""


@cli.command("ackley-99")
@click.argument("index_type", type=click.Choice(INDEX_TYPE_CHOICES))
def ackley_99(index_type: str) -> None:
    """TuRBO-ENN optimization-performance check on 99-D Ackley."""
    run_turbo_enn_ackley(index_driver=parse_index_driver(index_type))


@dataclass(frozen=True)
class YBoundsMetrics:
    name: str
    rmse: float
    mae: float
    nll: float
    frac_nonpos_mu: float
    frac_nonpos_samples: float
    frac_oob_samples: float


def y_bounds_array(lo: float, hi: float) -> np.ndarray:
    return np.array([[lo, hi]], dtype=float)


def bounds_label(lo: float, hi: float) -> str:
    def _tok(v: float) -> str:
        if v == np.inf:
            return "inf"
        if v == -np.inf:
            return "-inf"
        if v == int(v):
            return str(int(v))
        return f"{v:g}"

    return f"y_bounds_({_tok(lo)},{_tok(hi)})"


def inv_scalar_warp(z: float, lo: float, hi: float) -> float:
    """Inverse y-bounds warp (matches rust/crates/ennbo/src/y_bounds.rs)."""
    lo_fin = np.isfinite(lo)
    hi_fin = np.isfinite(hi)
    if not lo_fin and not hi_fin:
        return z
    if lo_fin and not hi_fin:
        return lo + float(np.exp(z))
    if not lo_fin and hi_fin:
        return hi - float(np.exp(-z))
    s = 1.0 / (1.0 + np.exp(-z))
    return lo + (hi - lo) * s


def _open_interval_margin(lo: float, hi: float) -> float:
    if np.isfinite(lo) and np.isfinite(hi):
        return 0.05 * (hi - lo)
    if np.isfinite(lo):
        return max(1e-3, 0.05 * abs(lo) if lo != 0.0 else 1e-3)
    if np.isfinite(hi):
        return max(1e-3, 0.05 * abs(hi) if hi != 0.0 else 1e-3)
    return 1.0


def clip_to_open_interval(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    margin = _open_interval_margin(lo, hi)
    out = np.asarray(y, dtype=float).copy()
    if np.isfinite(lo):
        out = np.maximum(out, lo + margin)
    if np.isfinite(hi):
        out = np.minimum(out, hi - margin)
    return out


def make_bounded_1d_xy(
    n: int,
    rng: np.random.Generator,
    lo: float,
    hi: float,
    *,
    y_scale: float = 1.0,
    y_center: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """1-d x on [0, 1]; y in open (lo, hi) via inverse warp of a smooth latent z."""
    x = rng.uniform(0.0, 1.0, size=(n, 1))
    noise = Y_BOUNDS_LOG_NOISE * rng.standard_normal(n)
    z = y_scale * np.sin(2.0 * np.pi * x[:, 0]) + y_center + noise
    z = np.clip(z, -12.0, 12.0)
    y = np.array(
        [inv_scalar_warp(float(zz), lo, hi) for zz in z],
        dtype=float,
    ).reshape(-1, 1)
    return x, clip_to_open_interval(y, lo, hi)


def make_positive_1d_xy(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """1-d x on [0, 1]; 1-d y in (0, inf)."""
    return make_bounded_1d_xy(n, rng, 0.0, np.inf)


def gaussian_nll(y: np.ndarray, mu: np.ndarray, se: np.ndarray) -> float:
    var = np.maximum(np.asarray(se, dtype=float) ** 2, 1e-12)
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + (y_arr - mu_arr) ** 2 / var))


def frac_out_of_open_interval(
    values: np.ndarray, lo: float, hi: float
) -> float:
    flat = np.asarray(values, dtype=float).ravel()
    out = np.zeros(flat.shape, dtype=bool)
    if np.isfinite(lo):
        out |= flat <= lo
    if np.isfinite(hi):
        out |= flat >= hi
    return float(np.mean(out))


def evaluate_natural_y(
    model: EpistemicNearestNeighbors,
    params: ENNParams,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    name: str,
    rng: np.random.Generator,
    num_draws: int,
    interval: tuple[float, float] | None = None,
) -> YBoundsMetrics:
    post = model.posterior(x_test, params=params)
    mu = np.asarray(post.mu, dtype=float).ravel()
    se = np.asarray(post.se, dtype=float).ravel()
    y = np.asarray(y_test, dtype=float).ravel()
    err = mu - y
    draws = post.sample(num_draws, rng)
    lo, hi = interval if interval is not None else (-np.inf, np.inf)
    return YBoundsMetrics(
        name=name,
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        nll=gaussian_nll(y, mu, se),
        frac_nonpos_mu=float(np.mean(mu <= 0.0)),
        frac_nonpos_samples=float(np.mean(draws <= 0.0)),
        frac_oob_samples=frac_out_of_open_interval(draws, lo, hi),
    )


def fit_and_eval_y_bounds(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    name: str,
    y_bounds: np.ndarray | None,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int,
    rng: np.random.Generator,
    num_draws: int,
    interval: tuple[float, float] | None = None,
) -> YBoundsMetrics:
    model = EpistemicNearestNeighbors(x_train, y_train, y_bounds=y_bounds)
    params = enn_fit(
        model,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        rng=rng,
    )
    return evaluate_natural_y(
        model,
        params,
        x_test,
        y_test,
        name=name,
        rng=rng,
        num_draws=num_draws,
        interval=interval,
    )


def compare_unbounded_vs_bounded(
    lo: float,
    hi: float,
    *,
    n_train: int = Y_BOUNDS_N_TRAIN,
    n_test: int = Y_BOUNDS_N_TEST,
    k: int = Y_BOUNDS_K,
    num_fit_candidates: int = Y_BOUNDS_NUM_FIT_CANDIDATES,
    num_fit_samples: int = Y_BOUNDS_NUM_FIT_SAMPLES,
    seed: int = SEED,
    num_draws: int = Y_BOUNDS_NUM_DRAWS,
    y_scale: float = 1.0,
    y_center: float = 0.0,
) -> tuple[YBoundsMetrics, YBoundsMetrics]:
    """Fit unbounded and matching-bounds ENNs on the same split; return both metrics."""
    data_rng = np.random.default_rng(seed)
    x_train, y_train = make_bounded_1d_xy(
        n_train, data_rng, lo, hi, y_scale=y_scale, y_center=y_center
    )
    x_test, y_test = make_bounded_1d_xy(
        n_test, data_rng, lo, hi, y_scale=y_scale, y_center=y_center
    )
    interval = (lo, hi)
    fit_rng = np.random.default_rng(seed + 1)
    unbounded = fit_and_eval_y_bounds(
        x_train,
        y_train,
        x_test,
        y_test,
        name="unbounded",
        y_bounds=None,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        rng=fit_rng,
        num_draws=num_draws,
        interval=interval,
    )
    bounded = fit_and_eval_y_bounds(
        x_train,
        y_train,
        x_test,
        y_test,
        name=bounds_label(lo, hi),
        y_bounds=y_bounds_array(lo, hi),
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        rng=np.random.default_rng(seed + 1),
        num_draws=num_draws,
        interval=interval,
    )
    return unbounded, bounded


def echo_y_bounds_metrics(metrics: YBoundsMetrics) -> None:
    click.echo(
        f"EVAL: model = {metrics.name} "
        f"SMALLER(rmse) = {metrics.rmse:.04f} "
        f"SMALLER(mae) = {metrics.mae:.04f} "
        f"SMALLER(nll) = {metrics.nll:.04f} "
        f"SMALLER(frac_nonpos_mu) = {metrics.frac_nonpos_mu:.04f} "
        f"SMALLER(frac_nonpos_samples) = {metrics.frac_nonpos_samples:.04f} "
        f"SMALLER(frac_oob_samples) = {metrics.frac_oob_samples:.04f}"
    )


def run_y_bounds(
    *,
    n_train: int = Y_BOUNDS_N_TRAIN,
    n_test: int = Y_BOUNDS_N_TEST,
    k: int = Y_BOUNDS_K,
    num_fit_candidates: int = Y_BOUNDS_NUM_FIT_CANDIDATES,
    num_fit_samples: int = Y_BOUNDS_NUM_FIT_SAMPLES,
    seed: int = SEED,
    num_draws: int = Y_BOUNDS_NUM_DRAWS,
) -> list[YBoundsMetrics]:
    """Fit unbounded and (0, inf) ENNs; score natural-y predictions on a test set."""
    click.echo(f"num_train={n_train} num_test={n_test} k={k}")
    unbounded, bounded = compare_unbounded_vs_bounded(
        0.0,
        np.inf,
        n_train=n_train,
        n_test=n_test,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        seed=seed,
        num_draws=num_draws,
    )
    results = [unbounded, bounded]
    for metrics in results:
        echo_y_bounds_metrics(metrics)
    return results


@cli.command("y-bounds")
@click.option("--n-train", type=int, default=Y_BOUNDS_N_TRAIN, show_default=True)
@click.option("--n-test", type=int, default=Y_BOUNDS_N_TEST, show_default=True)
@click.option("--k", type=int, default=Y_BOUNDS_K, show_default=True)
@click.option(
    "--num-fit-candidates",
    type=int,
    default=Y_BOUNDS_NUM_FIT_CANDIDATES,
    show_default=True,
)
@click.option(
    "--num-fit-samples",
    type=int,
    default=Y_BOUNDS_NUM_FIT_SAMPLES,
    show_default=True,
)
@click.option("--seed", type=int, default=SEED, show_default=True)
def y_bounds_cmd(
    n_train: int,
    n_test: int,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int,
    seed: int,
) -> None:
    """Compare unbounded vs (0, inf) y-bounds ENN on 1-d positive y."""
    run_y_bounds(
        n_train=n_train,
        n_test=n_test,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        seed=seed,
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
