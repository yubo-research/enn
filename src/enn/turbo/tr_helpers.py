from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine


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


def generate_tr_candidates(
    compute_bounds_1d: Any,
    x_center: np.ndarray,
    lengthscales: np.ndarray | None,
    num_candidates: int,
    *,
    rng: Generator,
    sobol_engine: QMCEngine,
) -> np.ndarray:
    from .turbo_utils import generate_raasp_candidates

    lb, ub = compute_bounds_1d(x_center, lengthscales)
    return generate_raasp_candidates(
        x_center, lb, ub, num_candidates, rng=rng, sobol_engine=sobol_engine
    )
