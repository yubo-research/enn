from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from . import turbo_optimizer_utils, turbo_utils
from .config.candidate_rv import CandidateRV

if TYPE_CHECKING:
    from numpy.random import Generator

    from .config.optimizer_config import OptimizerConfig


@dataclass(frozen=True)
class _CandidateGenContext:
    config: OptimizerConfig
    tr_state: Any
    num_dim: int
    sobol_seed_base: int
    restart_generation: int
    rng: Generator


def generate_optimizer_candidates(
    ctx: _CandidateGenContext,
    x_center: np.ndarray,
    lengthscales: np.ndarray | None,
    n_obs: int,
    *,
    num_arms: int,
) -> np.ndarray:
    if lengthscales is not None:
        lengthscales = np.asarray(lengthscales, dtype=float).reshape(-1)
        if not np.all(np.isfinite(lengthscales)):
            raise ValueError("lengthscales must be finite")
    num_candidates = int(
        ctx.config.candidates.num_candidates(num_dim=ctx.num_dim, num_arms=num_arms)
    )
    if num_candidates <= 0:
        raise ValueError(num_candidates)
    candidate_rv = ctx.config.candidate_rv
    if candidate_rv == CandidateRV.SOBOL:
        from scipy.stats import qmc

        sobol_seed = turbo_optimizer_utils.sobol_seed_for_state(
            ctx.sobol_seed_base,
            restart_generation=ctx.restart_generation,
            n_obs=n_obs,
            num_arms=num_arms,
        )
        sobol_engine = qmc.Sobol(d=ctx.num_dim, scramble=True, seed=sobol_seed)
    else:
        sobol_engine = None
    if getattr(ctx.tr_state, "uses_custom_candidate_gen", False):
        return ctx.tr_state.generate_candidates(
            x_center,
            lengthscales,
            num_candidates,
            rng=ctx.rng,
            sobol_engine=sobol_engine,
            raasp_driver=ctx.config.raasp_driver,
            num_pert=20,
        )
    return turbo_utils.generate_tr_candidates(
        ctx.tr_state.compute_bounds_1d,
        x_center,
        lengthscales,
        num_candidates,
        rng=ctx.rng,
        candidate_rv=candidate_rv,
        sobol_engine=sobol_engine,
        raasp_driver=ctx.config.raasp_driver,
        num_pert=20,
    )
