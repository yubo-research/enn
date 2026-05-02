from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .config.candidate_rv import CandidateRV
from .config.raasp_driver import RAASPDriver
from .turbo_utils_perturb import (
    generate_raasp_candidates,
    generate_raasp_candidates_uniform,
)

if TYPE_CHECKING:
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine

__all__ = [
    "generate_tr_candidates",
    "generate_tr_candidates_orig",
    "generate_tr_candidates_fast",
]


def _generate_tr_candidates_raasp(
    x_center: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    num_candidates: int,
    *,
    rng: Generator,
    candidate_rv: CandidateRV,
    sobol_engine: QMCEngine | None,
    num_pert: int,
) -> np.ndarray:
    if candidate_rv == CandidateRV.SOBOL and sobol_engine is None:
        raise ValueError("sobol_engine is required when candidate_rv=CandidateRV.SOBOL")
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
    lb, ub = compute_bounds_1d(x_center, lengthscales)
    _dispatch = {
        CandidateRV.SOBOL: lambda: _generate_tr_candidates_raasp(
            x_center,
            lb,
            ub,
            num_candidates,
            rng=rng,
            candidate_rv=candidate_rv,
            sobol_engine=sobol_engine,
            num_pert=num_pert,
        ),
        CandidateRV.UNIFORM: lambda: generate_raasp_candidates_uniform(
            x_center, lb, ub, num_candidates, rng=rng, num_pert=num_pert
        ),
        CandidateRV.RAASP: lambda: _generate_tr_candidates_raasp(
            x_center,
            lb,
            ub,
            num_candidates,
            rng=rng,
            candidate_rv=CandidateRV.RAASP,
            sobol_engine=sobol_engine,
            num_pert=num_pert,
        ),
    }
    if candidate_rv in _dispatch:
        return _dispatch[candidate_rv]()
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
