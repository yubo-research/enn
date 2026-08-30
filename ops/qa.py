#!/usr/bin/env python

from __future__ import annotations

import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass

import click
import numpy as np

from enn import create_optimizer, turbo_enn_config, turbo_zero_config
from enn.benchmarks import Ackley
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.enn_params import ENNParams, PosteriorFlags
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
INDEX_TYPE_CHOICES: tuple[str, ...] = ("flat", "fast_mem", "bpann_disk")
Y_BOUNDS_N_TRAIN = 80
Y_BOUNDS_N_TEST = 200
Y_BOUNDS_K = 10
Y_BOUNDS_NUM_FIT_CANDIDATES = 30
Y_BOUNDS_NUM_FIT_SAMPLES = 20
Y_BOUNDS_NUM_DRAWS = 64
Y_BOUNDS_LOG_NOISE = 0.15

TURBO_ACQ_NUM_DIM = 5
TURBO_ACQ_NUM_ROUNDS = 15
TURBO_ACQ_NUM_ARMS = 5
TURBO_ACQ_NUM_INIT = 5
TURBO_ACQ_NUM_SEEDS = 3
TURBO_ACQ_K = 5
TURBO_ACQ_NUM_FIT_SAMPLES = 20
TURBO_ACQ_NOISELESS_ACQS: tuple[AcqType, ...] = (
    AcqType.UCB,
    AcqType.THOMPSON,
    AcqType.PARETO,
)
TURBO_ACQ_NOISY_ACQS: tuple[AcqType, ...] = (AcqType.UCB, AcqType.THOMPSON)
TURBO_ACQ_NOISY_NOISE = 0.1

Y_VAR_N_TRAIN = 80
Y_VAR_N_TEST = 200
Y_VAR_K = 10
Y_VAR_NUM_FIT_CANDIDATES = 30
Y_VAR_NUM_FIT_SAMPLES = 20
Y_VAR_SIGMA = 0.3
Y_VAR_WRONG_SCALE = 0.01
Y_VAR_SWEEP: tuple[float, ...] = (0.01, 0.09, 0.25, 1.0)


def parse_index_driver(name: str) -> ENNIndexDriver:
    mapping = {
        "flat": ENNIndexDriver.FLAT,
        "fast_mem": ENNIndexDriver.FAST_MEM,
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


@dataclass(frozen=True)
class TurboAcqSeedResult:
    y_best: float
    time_seconds: float
    auc_y_best: float
    trajectory: tuple[float, ...]


@dataclass(frozen=True)
class TurboAcqMetrics:
    problem: str
    method: str
    y_best_mean: float
    y_best_se: float
    time_seconds: float
    auc_y_best: float


def auc_of_y_best_trajectory(trajectory: Sequence[float]) -> float:
    """Normalized trapezoidal AUC of the running y_best curve."""
    if not trajectory:
        return float("nan")
    ys = np.asarray(trajectory, dtype=float)
    if len(ys) == 1:
        return float(ys[0])
    xs = np.arange(len(ys), dtype=float)
    trapz = getattr(np, "trapezoid", None)
    if trapz is None:
        trapz = np.trapz
    return float(trapz(ys, xs) / xs[-1])


def build_turbo_acq_config(
    *,
    acq_type: AcqType | None,
    noise: float,
    num_init: int = TURBO_ACQ_NUM_INIT,
    k: int = TURBO_ACQ_K,
    num_fit_samples: int = TURBO_ACQ_NUM_FIT_SAMPLES,
    turbo_zero: bool = False,
):
    """Build TuRBO-ENN or TuRBO-ZERO config for the acquisition sweep."""
    noise_aware = float(noise) > 0.0
    trust = TurboTRConfig(noise_aware=noise_aware)
    if turbo_zero:
        return turbo_zero_config(
            num_init=num_init,
            trust_region=trust,
            candidate_rv=CandidateRV.UNIFORM,
        )
    if acq_type is None:
        raise ValueError("acq_type is required unless turbo_zero=True")
    return turbo_enn_config(
        enn=ENNSurrogateConfig(
            k=k,
            fit=ENNFitConfig(num_fit_samples=num_fit_samples),
        ),
        candidates=CandidateGenConfig(candidate_rv=CandidateRV.UNIFORM),
        trust_region=trust,
        acq_type=acq_type,
        num_init=num_init,
    )


def run_turbo_acq_seed(
    *,
    acq_type: AcqType | None,
    noise: float,
    seed: int,
    num_dim: int = TURBO_ACQ_NUM_DIM,
    num_rounds: int = TURBO_ACQ_NUM_ROUNDS,
    num_arms: int = TURBO_ACQ_NUM_ARMS,
    num_init: int = TURBO_ACQ_NUM_INIT,
    k: int = TURBO_ACQ_K,
    num_fit_samples: int = TURBO_ACQ_NUM_FIT_SAMPLES,
    turbo_zero: bool = False,
) -> TurboAcqSeedResult:
    """One seed of TuRBO ask/tell on Ackley; returns y_best, time, and AUC."""
    obj_rng = np.random.default_rng(seed)
    objective = Ackley(noise=noise, rng=obj_rng)
    bounds = np.array([objective.bounds] * num_dim, dtype=float)
    config = build_turbo_acq_config(
        acq_type=acq_type,
        noise=noise,
        num_init=num_init,
        k=k,
        num_fit_samples=num_fit_samples,
        turbo_zero=turbo_zero,
    )
    optimizer = create_optimizer(
        bounds=bounds,
        config=config,
        rng=np.random.default_rng(seed),
    )
    y_var_scale = float(noise) ** 2
    y_best = -np.inf
    trajectory: list[float] = []
    t0 = time.perf_counter()
    for _ in range(num_rounds):
        x_arms = optimizer.ask(num_arms=num_arms)
        y_obs = np.asarray(objective(x_arms), dtype=float).reshape(-1, 1)
        if noise > 0.0:
            optimizer.tell(x_arms, y_obs, y_var=y_var_scale * np.ones_like(y_obs))
        else:
            optimizer.tell(x_arms, y_obs)
        y_best = max(y_best, float(np.max(y_obs)))
        trajectory.append(y_best)
    return TurboAcqSeedResult(
        y_best=y_best,
        time_seconds=time.perf_counter() - t0,
        auc_y_best=auc_of_y_best_trajectory(trajectory),
        trajectory=tuple(trajectory),
    )


def aggregate_turbo_acq_seeds(
    results: Sequence[TurboAcqSeedResult],
    *,
    problem: str,
    method: str,
) -> TurboAcqMetrics:
    y_bests = np.asarray([r.y_best for r in results], dtype=float)
    times = np.asarray([r.time_seconds for r in results], dtype=float)
    aucs = np.asarray([r.auc_y_best for r in results], dtype=float)
    n = len(results)
    se = float(np.std(y_bests, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return TurboAcqMetrics(
        problem=problem,
        method=method,
        y_best_mean=float(np.mean(y_bests)),
        y_best_se=se,
        time_seconds=float(np.mean(times)),
        auc_y_best=float(np.mean(aucs)),
    )


def echo_turbo_acq_metrics(metrics: TurboAcqMetrics) -> None:
    click.echo(
        f"EVAL: problem = {metrics.problem} method = {metrics.method} "
        f"LARGER(y_best_mean) = {metrics.y_best_mean:.04f} "
        f"SMALLER(y_best_se) = {metrics.y_best_se:.04f} "
        f"SMALLER(time_seconds) = {metrics.time_seconds:.04f} "
        f"LARGER(auc_y_best) = {metrics.auc_y_best:.04f}"
    )


def _method_label(acq_type: AcqType | None, *, turbo_zero: bool) -> str:
    if turbo_zero:
        return "turbo_zero"
    assert acq_type is not None
    return acq_type.value


def sweep_turbo_acq(
    *,
    problem: str,
    noise: float,
    acq_types: Sequence[AcqType],
    include_turbo_zero: bool,
    num_dim: int,
    num_rounds: int,
    num_arms: int,
    num_init: int,
    num_seeds: int,
    seed: int,
    k: int,
    num_fit_samples: int,
) -> list[TurboAcqMetrics]:
    """Run acquisition methods over seeds; return aggregated metrics."""
    methods: list[tuple[AcqType | None, bool]] = [
        (acq, False) for acq in acq_types
    ]
    if include_turbo_zero:
        methods.append((None, True))
    out: list[TurboAcqMetrics] = []
    for acq_type, turbo_zero in methods:
        seed_results = [
            run_turbo_acq_seed(
                acq_type=acq_type,
                noise=noise,
                seed=seed + i,
                num_dim=num_dim,
                num_rounds=num_rounds,
                num_arms=num_arms,
                num_init=num_init,
                k=k,
                num_fit_samples=num_fit_samples,
                turbo_zero=turbo_zero,
            )
            for i in range(num_seeds)
        ]
        metrics = aggregate_turbo_acq_seeds(
            seed_results,
            problem=problem,
            method=_method_label(acq_type, turbo_zero=turbo_zero),
        )
        out.append(metrics)
    return out


def run_turbo_acq(
    *,
    num_dim: int = TURBO_ACQ_NUM_DIM,
    num_rounds: int = TURBO_ACQ_NUM_ROUNDS,
    num_arms: int = TURBO_ACQ_NUM_ARMS,
    num_init: int = TURBO_ACQ_NUM_INIT,
    num_seeds: int = TURBO_ACQ_NUM_SEEDS,
    seed: int = SEED,
    k: int = TURBO_ACQ_K,
    num_fit_samples: int = TURBO_ACQ_NUM_FIT_SAMPLES,
    include_noisy: bool = True,
    include_turbo_zero: bool = True,
    noisy_noise: float = TURBO_ACQ_NOISY_NOISE,
) -> list[TurboAcqMetrics]:
    """Sweep acquisition types on noiseless Ackley; optional noisy + TuRBO-ZERO."""
    click.echo(
        f"num_dim={num_dim} num_rounds={num_rounds} num_arms={num_arms} "
        f"num_seeds={num_seeds} include_noisy={include_noisy} "
        f"include_turbo_zero={include_turbo_zero}"
    )
    results = sweep_turbo_acq(
        problem="noiseless",
        noise=0.0,
        acq_types=TURBO_ACQ_NOISELESS_ACQS,
        include_turbo_zero=include_turbo_zero,
        num_dim=num_dim,
        num_rounds=num_rounds,
        num_arms=num_arms,
        num_init=num_init,
        num_seeds=num_seeds,
        seed=seed,
        k=k,
        num_fit_samples=num_fit_samples,
    )
    if include_noisy:
        results.extend(
            sweep_turbo_acq(
                problem="noisy",
                noise=noisy_noise,
                acq_types=TURBO_ACQ_NOISY_ACQS,
                include_turbo_zero=False,
                num_dim=num_dim,
                num_rounds=num_rounds,
                num_arms=num_arms,
                num_init=num_init,
                num_seeds=num_seeds,
                seed=seed + 1000,
                k=k,
                num_fit_samples=num_fit_samples,
            )
        )
    for metrics in results:
        echo_turbo_acq_metrics(metrics)
    return results


@cli.command("turbo-acq")
@click.option("--num-dim", type=int, default=TURBO_ACQ_NUM_DIM, show_default=True)
@click.option("--num-rounds", type=int, default=TURBO_ACQ_NUM_ROUNDS, show_default=True)
@click.option("--num-arms", type=int, default=TURBO_ACQ_NUM_ARMS, show_default=True)
@click.option("--num-init", type=int, default=TURBO_ACQ_NUM_INIT, show_default=True)
@click.option("--num-seeds", type=int, default=TURBO_ACQ_NUM_SEEDS, show_default=True)
@click.option("--seed", type=int, default=SEED, show_default=True)
@click.option("--no-noisy", "include_noisy", is_flag=True, flag_value=False, default=True)
@click.option(
    "--no-turbo-zero",
    "include_turbo_zero",
    is_flag=True,
    flag_value=False,
    default=True,
)
def turbo_acq_cmd(
    num_dim: int,
    num_rounds: int,
    num_arms: int,
    num_init: int,
    num_seeds: int,
    seed: int,
    include_noisy: bool,
    include_turbo_zero: bool,
) -> None:
    """Compare TuRBO acquisition types on Ackley (noiseless primary)."""
    run_turbo_acq(
        num_dim=num_dim,
        num_rounds=num_rounds,
        num_arms=num_arms,
        num_init=num_init,
        num_seeds=num_seeds,
        seed=seed,
        include_noisy=include_noisy,
        include_turbo_zero=include_turbo_zero,
    )


@dataclass(frozen=True)
class YVarNoiseMetrics:
    name: str
    nll: float
    calib_1se: float
    mean_se: float
    mean_se_ale: float


def make_noisy_1d_regression(
    n: int,
    rng: np.random.Generator,
    *,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-d x on [-1, 1]; noisy sine observations; returns x, y, f(x)."""
    x = rng.uniform(-1.0, 1.0, size=(n, 1))
    f = np.sin(2.0 * np.pi * x)
    y = f + sigma * rng.standard_normal(size=(n, 1))
    return x, y, f


def evaluate_y_var_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    name: str,
    train_yvar: np.ndarray | None,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int,
    rng: np.random.Generator,
) -> YVarNoiseMetrics:
    model = EpistemicNearestNeighbors(x_train, y_train, train_yvar=train_yvar)
    params = enn_fit(
        model,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        rng=rng,
    )
    flags = PosteriorFlags(observation_noise=True)
    post = model.posterior(x_test, params=params, flags=flags)
    mu = np.asarray(post.mu, dtype=float).ravel()
    se = np.asarray(post.se, dtype=float).ravel()
    se_ale = np.asarray(post.se_ale, dtype=float).ravel()
    y = np.asarray(y_test, dtype=float).ravel()
    return YVarNoiseMetrics(
        name=name,
        nll=gaussian_nll(y, mu, se),
        calib_1se=float(np.mean(np.abs(y - mu) <= se)),
        mean_se=float(np.mean(se)),
        mean_se_ale=float(np.mean(se_ale)),
    )


def compare_y_var_noise(
    *,
    n_train: int = Y_VAR_N_TRAIN,
    n_test: int = Y_VAR_N_TEST,
    k: int = Y_VAR_K,
    num_fit_candidates: int = Y_VAR_NUM_FIT_CANDIDATES,
    num_fit_samples: int = Y_VAR_NUM_FIT_SAMPLES,
    seed: int = SEED,
    sigma: float = Y_VAR_SIGMA,
    wrong_scale: float = Y_VAR_WRONG_SCALE,
) -> tuple[YVarNoiseMetrics, YVarNoiseMetrics, YVarNoiseMetrics]:
    """Fit matched / none / wrong yvar models on the same noisy split."""
    data_rng = np.random.default_rng(seed)
    x_train, y_train, _ = make_noisy_1d_regression(n_train, data_rng, sigma=sigma)
    x_test, y_test, _ = make_noisy_1d_regression(n_test, data_rng, sigma=sigma)
    matched_yvar = (sigma**2) * np.ones_like(y_train)
    wrong_yvar = wrong_scale * np.ones_like(y_train)
    fit_seed = seed + 1
    matched = evaluate_y_var_fit(
        x_train,
        y_train,
        x_test,
        y_test,
        name="matched",
        train_yvar=matched_yvar,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        rng=np.random.default_rng(fit_seed),
    )
    none_m = evaluate_y_var_fit(
        x_train,
        y_train,
        x_test,
        y_test,
        name="none",
        train_yvar=None,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        rng=np.random.default_rng(fit_seed),
    )
    wrong = evaluate_y_var_fit(
        x_train,
        y_train,
        x_test,
        y_test,
        name="wrong",
        train_yvar=wrong_yvar,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        rng=np.random.default_rng(fit_seed),
    )
    return matched, none_m, wrong


def sweep_y_var_se_ale(
    *,
    n_train: int = Y_VAR_N_TRAIN,
    n_test: int = Y_VAR_N_TEST,
    k: int = Y_VAR_K,
    num_fit_candidates: int = Y_VAR_NUM_FIT_CANDIDATES,
    num_fit_samples: int = Y_VAR_NUM_FIT_SAMPLES,
    seed: int = SEED,
    sigma: float = Y_VAR_SIGMA,
    yvar_scales: Sequence[float] = Y_VAR_SWEEP,
) -> list[tuple[float, float]]:
    """Mean se_ale vs supplied constant yvar scale (monotonicity probe)."""
    data_rng = np.random.default_rng(seed)
    x_train, y_train, _ = make_noisy_1d_regression(n_train, data_rng, sigma=sigma)
    x_test, y_test, _ = make_noisy_1d_regression(n_test, data_rng, sigma=sigma)
    out: list[tuple[float, float]] = []
    for scale in yvar_scales:
        metrics = evaluate_y_var_fit(
            x_train,
            y_train,
            x_test,
            y_test,
            name=f"yvar_{scale:g}",
            train_yvar=float(scale) * np.ones_like(y_train),
            k=k,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
            rng=np.random.default_rng(seed + 1),
        )
        out.append((float(scale), metrics.mean_se_ale))
    return out


def echo_y_var_noise_metrics(metrics: YVarNoiseMetrics) -> None:
    click.echo(
        f"EVAL: model = {metrics.name} "
        f"SMALLER(nll) = {metrics.nll:.04f} "
        f"LARGER(calib_1se) = {metrics.calib_1se:.04f} "
        f"mean_se = {metrics.mean_se:.04f} "
        f"mean_se_ale = {metrics.mean_se_ale:.04f}"
    )


def echo_y_var_se_ale_point(scale: float, mean_se_ale: float) -> None:
    click.echo(
        f"EVAL: yvar_scale = {scale:.04f} "
        f"LARGER(mean_se_ale) = {mean_se_ale:.04f}"
    )


def run_y_var_noise(
    *,
    n_train: int = Y_VAR_N_TRAIN,
    n_test: int = Y_VAR_N_TEST,
    k: int = Y_VAR_K,
    num_fit_candidates: int = Y_VAR_NUM_FIT_CANDIDATES,
    num_fit_samples: int = Y_VAR_NUM_FIT_SAMPLES,
    seed: int = SEED,
    sigma: float = Y_VAR_SIGMA,
    wrong_scale: float = Y_VAR_WRONG_SCALE,
    yvar_scales: Sequence[float] = Y_VAR_SWEEP,
) -> tuple[list[YVarNoiseMetrics], list[tuple[float, float]]]:
    """Compare matched vs mismatched train_yvar; probe se_ale vs yvar scale."""
    click.echo(
        f"num_train={n_train} num_test={n_test} k={k} sigma={sigma:.04f}"
    )
    matched, none_m, wrong = compare_y_var_noise(
        n_train=n_train,
        n_test=n_test,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        seed=seed,
        sigma=sigma,
        wrong_scale=wrong_scale,
    )
    models = [matched, none_m, wrong]
    for metrics in models:
        echo_y_var_noise_metrics(metrics)
    sweep = sweep_y_var_se_ale(
        n_train=n_train,
        n_test=n_test,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        seed=seed,
        sigma=sigma,
        yvar_scales=yvar_scales,
    )
    for scale, mean_se_ale in sweep:
        echo_y_var_se_ale_point(scale, mean_se_ale)
    return models, sweep


@cli.command("y-var-noise")
@click.option("--n-train", type=int, default=Y_VAR_N_TRAIN, show_default=True)
@click.option("--n-test", type=int, default=Y_VAR_N_TEST, show_default=True)
@click.option("--k", type=int, default=Y_VAR_K, show_default=True)
@click.option(
    "--num-fit-candidates",
    type=int,
    default=Y_VAR_NUM_FIT_CANDIDATES,
    show_default=True,
)
@click.option(
    "--num-fit-samples",
    type=int,
    default=Y_VAR_NUM_FIT_SAMPLES,
    show_default=True,
)
@click.option("--seed", type=int, default=SEED, show_default=True)
@click.option("--sigma", type=float, default=Y_VAR_SIGMA, show_default=True)
def y_var_noise_cmd(
    n_train: int,
    n_test: int,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int,
    seed: int,
    sigma: float,
) -> None:
    """Compare matched vs mismatched observation noise (train_yvar)."""
    run_y_var_noise(
        n_train=n_train,
        n_test=n_test,
        k=k,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        seed=seed,
        sigma=sigma,
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
