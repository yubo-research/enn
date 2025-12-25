from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:
    import torch
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine

from .turbo_gp_fit import fit_gp
from .turbo_optimizer_utils import (
    sobol_seed_for_state,
    trim_trailing_observations,
    validate_tell_inputs,
)

__all__ = [
    "Telemetry",
    "fit_gp",
    "sobol_seed_for_state",
    "trim_trailing_observations",
    "validate_tell_inputs",
]


@dataclass(frozen=True)
class Telemetry:
    dt_fit: float
    dt_sel: float


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


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
    except Exception:  # pragma: no cover
        GPInputWarning = None

    if GPInputWarning is None:
        return model.posterior(x_torch)
    # We intentionally evaluate the GP posterior at the training inputs
    # (observed points) when choosing the center. GPyTorch warns about this
    # in debug mode, but it's expected for our usage.
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
    x = np.zeros((num_points, num_dim))
    centers = (1.0 + 2.0 * np.arange(0.0, num_points)) / float(2 * num_points)
    for j in range(num_dim):
        x[:, j] = centers[rng.permutation(num_points)]
    pert = rng.uniform(-1.0, 1.0, size=(num_points, num_dim)) / float(2 * num_points)
    x += pert
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
    n_sobol = _next_power_of_2(num_candidates)
    sobol_samples = sobol_engine.random(n_sobol)[:num_candidates]
    lb_array = np.asarray(lb)
    ub_array = np.asarray(ub)
    pert = lb_array + (ub_array - lb_array) * sobol_samples
    candidates = np.tile(x_center, (num_candidates, 1))
    if np.any(mask):
        candidates[mask] = pert[mask]
    return candidates


def raasp(
    x_center: np.ndarray | Any,
    lb: np.ndarray | list[float] | Any,
    ub: np.ndarray | list[float] | Any,
    num_candidates: int,
    *,
    num_pert: int = 20,
    rng: Generator | Any,
    sobol_engine: QMCEngine | Any,
) -> np.ndarray:
    num_dim = x_center.shape[-1]
    prob_perturb = min(num_pert / num_dim, 1.0)
    mask = rng.random((num_candidates, num_dim)) <= prob_perturb
    ind = np.nonzero(~mask.any(axis=1))[0]
    if len(ind) > 0:
        mask[ind, rng.integers(0, num_dim, size=len(ind))] = True
    return sobol_perturb_np(
        x_center, lb, ub, num_candidates, mask, sobol_engine=sobol_engine
    )


def generate_raasp_candidates(
    center: np.ndarray | Any,
    lb: np.ndarray | list[float] | Any,
    ub: np.ndarray | list[float] | Any,
    num_candidates: int,
    *,
    rng: Generator | Any,
    sobol_engine: QMCEngine | Any,
    num_pert: int = 20,
) -> np.ndarray:
    if num_candidates <= 0:
        raise ValueError(num_candidates)
    return raasp(
        center,
        lb,
        ub,
        num_candidates,
        num_pert=num_pert,
        rng=rng,
        sobol_engine=sobol_engine,
    )


def generate_trust_region_candidates(
    x_center: np.ndarray | Any,
    lengthscales: np.ndarray | None,
    num_candidates: int,
    *,
    compute_bounds_1d: Any,
    rng: Generator | Any,
    sobol_engine: QMCEngine | Any,
) -> np.ndarray:
    """
    Small DRY helper for trust-region candidate generation.

    `compute_bounds_1d` is typically a TR object's bound computation method.
    """
    lb, ub = compute_bounds_1d(x_center, lengthscales)
    return generate_raasp_candidates(
        x_center, lb, ub, num_candidates, rng=rng, sobol_engine=sobol_engine
    )


def validate_trust_region_request(
    num_arms: int, configured_num_arms: int, *, is_fallback: bool = False
) -> None:
    if is_fallback:
        if num_arms > configured_num_arms:
            raise ValueError(
                f"num_arms {num_arms} > configured num_arms {configured_num_arms}"
            )
    else:
        if num_arms != configured_num_arms:
            raise ValueError(
                f"num_arms {num_arms} != configured num_arms {configured_num_arms}"
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
