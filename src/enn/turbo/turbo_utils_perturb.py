from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .config.candidate_rv import CandidateRV

if TYPE_CHECKING:
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine

__all__ = [
    "latin_hypercube",
    "argmax_random_tie",
    "sobol_perturb_np",
    "uniform_perturb_np",
    "raasp_perturb",
    "generate_raasp_candidates",
    "generate_raasp_candidates_uniform",
    "to_unit",
    "from_unit",
]


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
    ks = np.maximum(rng.binomial(num_dim, prob_perturb, size=num_candidates), 1)
    mask = np.zeros((num_candidates, num_dim), dtype=bool)
    for i in range(num_candidates):
        idx = rng.choice(num_dim, size=ks[i], replace=False)
        mask[i, idx] = True

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
