from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING, Any, Callable, Iterator
import numpy as np

if TYPE_CHECKING:
    import torch
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine
    from .config.candidate_rv import CandidateRV
from .config.raasp_driver import RAASPDriver


@contextlib.contextmanager
def record_duration(set_dt: Callable[[float], None]) -> Iterator[None]:
    import time

    t0 = time.perf_counter()
    try:
        yield
    finally:
        set_dt(time.perf_counter() - t0)


@contextlib.contextmanager
def torch_seed_context(
    seed: int, device: torch.device | Any | None = None
) -> Iterator[None]:
    import torch

    devices: list[int] | None = None
    if device is not None and getattr(device, "type", None) == "cuda":
        idx = 0 if getattr(device, "index", None) is None else int(device.index)
        devices = [idx]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        if device is not None and getattr(device, "type", None) == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(int(seed))
        yield


def get_gp_posterior_suppress_warning(model: Any, x_torch: Any) -> Any:
    import warnings

    try:
        from gpytorch.utils.warnings import GPInputWarning
    except Exception:
        GPInputWarning = None
    if GPInputWarning is None:
        return model.posterior(x_torch)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The input matches the stored training data\..*",
            category=GPInputWarning,
        )
        return model.posterior(x_torch)


def latin_hypercube(
    num_points: int, num_dim: int, *, rng: Generator | Any
) -> np.ndarray:
    x = (1.0 + 2.0 * np.arange(0.0, num_points)) / float(2 * num_points)
    x = np.stack([x[rng.permutation(num_points)] for _ in range(num_dim)], axis=1)
    x += rng.uniform(-1.0, 1.0, size=(num_points, num_dim)) / float(2 * num_points)
    assert x.shape == (num_points, num_dim)
    return x


def argmax_random_tie(values: np.ndarray | Any, *, rng: Generator | Any) -> int:
    if values.ndim != 1:
        raise ValueError(values.shape)
    max_val = float(np.max(values))
    idx = np.nonzero(values >= max_val)[0]
    if idx.size == 0:
        return int(rng.integers(values.size))
    if idx.size == 1:
        return int(idx[0])
    j = int(rng.integers(idx.size))
    return int(idx[j])


def sobol_perturb_np(
    x_center: np.ndarray | Any,
    lb: np.ndarray | list[float] | Any,
    ub: np.ndarray | list[float] | Any,
    num_candidates: int,
    mask: np.ndarray | Any,
    *,
    sobol_engine: QMCEngine | Any,
) -> np.ndarray:
    n = num_candidates
    n_sobol = 1 if n <= 0 else 1 << (n - 1).bit_length()
    sobol_samples = sobol_engine.random(n_sobol)[:num_candidates]
    lb_array = np.asarray(lb)
    ub_array = np.asarray(ub)
    pert = lb_array + (ub_array - lb_array) * sobol_samples
    candidates = np.tile(x_center, (num_candidates, 1))
    if np.any(mask):
        candidates[mask] = pert[mask]
    return candidates


def uniform_perturb_np(
    x_center: np.ndarray | Any,
    lb: np.ndarray | list[float] | Any,
    ub: np.ndarray | list[float] | Any,
    num_candidates: int,
    mask: np.ndarray | Any,
    *,
    rng: Generator | Any,
) -> np.ndarray:
    lb_array = np.asarray(lb)
    ub_array = np.asarray(ub)
    pert = lb_array + (ub_array - lb_array) * rng.uniform(
        0.0, 1.0, size=(num_candidates, x_center.shape[-1])
    )
    candidates = np.tile(x_center, (num_candidates, 1))
    if np.any(mask):
        candidates[mask] = pert[mask]
    return candidates


def raasp_perturb(
    x_center: np.ndarray | Any,
    lb: np.ndarray | list[float] | Any,
    ub: np.ndarray | list[float] | Any,
    num_candidates: int,
    *,
    num_pert: int = 20,
    rng: Generator | Any,
    candidate_rv: CandidateRV,
    sobol_engine: QMCEngine | Any | None = None,
) -> np.ndarray:
    num_dim = x_center.shape[-1]
    prob_perturb = min(num_pert / num_dim, 1.0)
    # Use binomial to determine how many dimensions to perturb for each candidate
    ks = np.maximum(rng.binomial(num_dim, prob_perturb, size=num_candidates), 1)
    mask = np.zeros((num_candidates, num_dim), dtype=bool)
    for i in range(num_candidates):
        idx = rng.choice(num_dim, size=ks[i], replace=False)
        mask[i, idx] = True

    from .config.candidate_rv import CandidateRV

    if candidate_rv == CandidateRV.SOBOL:
        if sobol_engine is None:
            raise ValueError("sobol_engine required for CandidateRV.SOBOL")
        return sobol_perturb_np(
            x_center, lb, ub, num_candidates, mask, sobol_engine=sobol_engine
        )
    return uniform_perturb_np(x_center, lb, ub, num_candidates, mask, rng=rng)


def generate_raasp_candidates(
    center: np.ndarray | Any,
    lb: np.ndarray | list[float] | Any,
    ub: np.ndarray | list[float] | Any,
    num_candidates: int,
    *,
    rng: Generator | Any,
    candidate_rv: CandidateRV,
    sobol_engine: QMCEngine | Any | None = None,
    num_pert: int = 20,
) -> np.ndarray:
    if num_candidates <= 0:
        raise ValueError(num_candidates)
    return raasp_perturb(
        center,
        lb,
        ub,
        num_candidates,
        num_pert=num_pert,
        rng=rng,
        candidate_rv=candidate_rv,
        sobol_engine=sobol_engine,
    )


def generate_raasp_candidates_uniform(
    center: np.ndarray | Any,
    lb: np.ndarray | list[float] | Any,
    ub: np.ndarray | list[float] | Any,
    num_candidates: int,
    *,
    rng: Generator | Any,
    num_pert: int = 20,
) -> np.ndarray:
    from .config.candidate_rv import CandidateRV

    return generate_raasp_candidates(
        center,
        lb,
        ub,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
        num_pert=num_pert,
    )


def to_unit(x: np.ndarray | Any, bounds: np.ndarray | Any) -> np.ndarray:
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    if np.any(ub <= lb):
        raise ValueError(bounds)
    return (x - lb) / (ub - lb)


def from_unit(x_unit: np.ndarray | Any, bounds: np.ndarray | Any) -> np.ndarray:
    lb = np.asarray(bounds[:, 0])
    ub = np.asarray(bounds[:, 1])
    return lb + x_unit * (ub - lb)


def gp_thompson_sample(
    model: Any,
    x_cand: np.ndarray | Any,
    num_arms: int,
    rng: Generator | Any,
    *,
    gp_y_mean: float,
    gp_y_std: float,
) -> np.ndarray:
    import gpytorch
    import torch

    x_torch = torch.as_tensor(x_cand, dtype=torch.float64)
    seed = int(rng.integers(2**31 - 1))
    with (
        torch.no_grad(),
        gpytorch.settings.fast_pred_var(),
        torch_seed_context(seed, device=x_torch.device),
    ):
        posterior = model.posterior(x_torch)
        samples = posterior.sample(sample_shape=torch.Size([1]))
    if samples.ndim != 2:
        raise ValueError(samples.shape)
    ts = samples[0].reshape(-1)
    scores = ts.detach().cpu().numpy().reshape(-1)
    scores = gp_y_mean + gp_y_std * scores
    shuffled_indices = rng.permutation(len(scores))
    shuffled_scores = scores[shuffled_indices]
    top_k_in_shuffled = np.argpartition(-shuffled_scores, num_arms - 1)[:num_arms]
    idx = shuffled_indices[top_k_in_shuffled]
    return idx


def compute_full_box_bounds_1d(
    x_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lb = np.zeros_like(x_center, dtype=float)
    ub = np.ones_like(x_center, dtype=float)
    return lb, ub


def get_single_incumbent_index(
    selector: Any,
    y: np.ndarray,
    rng: Generator,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.array([], dtype=int)
    best_idx = selector.select(y, mu, rng)
    return np.array([best_idx])


def get_incumbent_index(
    selector: Any,
    y: np.ndarray,
    rng: Generator,
    mu: np.ndarray | None = None,
) -> int:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        raise ValueError("y is empty")
    return int(selector.select(y, mu, rng))


def get_scalar_incumbent_value(
    selector: Any,
    y_obs: np.ndarray,
    rng: Generator,
    *,
    mu_obs: np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(y_obs, dtype=float)
    if y.size == 0:
        return np.array([], dtype=float)
    idx = get_incumbent_index(selector, y, rng, mu=mu_obs)
    use_mu = bool(getattr(selector, "noise_aware", False))
    values = mu_obs if use_mu else y
    if values is None:
        raise ValueError("noise_aware incumbent selection requires mu_obs")
    v = np.asarray(values, dtype=float)
    if v.ndim == 2:
        value = float(v[idx, 0])
    elif v.ndim == 1:
        value = float(v[idx])
    else:
        raise ValueError(v.shape)
    return np.array([value], dtype=float)


class ScalarIncumbentMixin:
    incumbent_selector: Any

    def get_incumbent_index(
        self,
        y: np.ndarray | Any,
        rng: Generator,
        mu: np.ndarray | None = None,
    ) -> int:
        return get_incumbent_index(self.incumbent_selector, y, rng, mu=mu)

    def get_incumbent_value(
        self,
        y_obs: np.ndarray | Any,
        rng: Generator,
        mu_obs: np.ndarray | None = None,
    ) -> np.ndarray:
        return get_scalar_incumbent_value(
            self.incumbent_selector, y_obs, rng, mu_obs=mu_obs
        )


def generate_tr_candidates_orig(
    compute_bounds_1d: Any,
    x_center: np.ndarray,
    lengthscales: np.ndarray | None,
    num_candidates: int,
    *,
    rng: Generator,
    candidate_rv: CandidateRV,
    sobol_engine: QMCEngine | None = None,
    num_pert: int = 20,
) -> np.ndarray:
    from .config.candidate_rv import CandidateRV

    lb, ub = compute_bounds_1d(x_center, lengthscales)
    if candidate_rv == CandidateRV.SOBOL:
        if sobol_engine is None:
            raise ValueError(
                "sobol_engine is required when candidate_rv=CandidateRV.SOBOL"
            )
        return generate_raasp_candidates(
            x_center,
            lb,
            ub,
            num_candidates,
            rng=rng,
            candidate_rv=candidate_rv,
            sobol_engine=sobol_engine,
            num_pert=num_pert,
        )
    if candidate_rv == CandidateRV.UNIFORM:
        return generate_raasp_candidates_uniform(
            x_center, lb, ub, num_candidates, rng=rng, num_pert=num_pert
        )
    raise ValueError(candidate_rv)


def generate_tr_candidates_fast(
    compute_bounds_1d: Any,
    x_center: np.ndarray,
    lengthscales: np.ndarray | None,
    num_candidates: int,
    *,
    rng: Generator,
    candidate_rv: CandidateRV,
    num_pert: int,
) -> np.ndarray:
    from scipy.stats import qmc
    from .config.candidate_rv import CandidateRV

    lb, ub = compute_bounds_1d(x_center, lengthscales)
    num_dim = x_center.shape[-1]
    candidates = np.tile(x_center, (num_candidates, 1))
    prob_perturb = min(num_pert / num_dim, 1.0)
    ks = np.maximum(rng.binomial(num_dim, prob_perturb, size=num_candidates), 1)
    max_k = int(np.max(ks))
    samples = (
        qmc.Sobol(d=max_k, scramble=True, seed=int(rng.integers(0, 2**31))).random(
            num_candidates
        )
        if candidate_rv == CandidateRV.SOBOL
        else rng.random((num_candidates, max_k))
    )
    for i in range(num_candidates):
        idx = rng.choice(num_dim, size=ks[i], replace=False)
        candidates[i, idx] = lb[idx] + (ub[idx] - lb[idx]) * samples[i, : ks[i]]
    assert candidates.shape == (num_candidates, num_dim)
    return candidates


def generate_tr_candidates(
    compute_bounds_1d: Any,
    x_center: np.ndarray,
    lengthscales: np.ndarray | None,
    num_candidates: int,
    *,
    rng: Generator,
    candidate_rv: CandidateRV,
    sobol_engine: QMCEngine | None,
    raasp_driver: RAASPDriver,
    num_pert: int,
) -> np.ndarray:
    from .config.raasp_driver import RAASPDriver

    if raasp_driver == RAASPDriver.FAST:
        return generate_tr_candidates_fast(
            compute_bounds_1d,
            x_center,
            lengthscales,
            num_candidates,
            rng=rng,
            candidate_rv=candidate_rv,
            num_pert=num_pert,
        )
    return generate_tr_candidates_orig(
        compute_bounds_1d,
        x_center,
        lengthscales,
        num_candidates,
        rng=rng,
        candidate_rv=candidate_rv,
        sobol_engine=sobol_engine,
        num_pert=num_pert,
    )
