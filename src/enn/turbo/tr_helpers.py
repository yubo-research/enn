from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .config.enums import CandidateRV

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine


def compute_full_box_bounds_1d(
    x_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    lb = np.zeros_like(x_center, dtype=float)
    ub = np.ones_like(x_center, dtype=float)
    return lb, ub


def get_single_incumbent_index(
    selector,
    y: np.ndarray,
    rng: Generator,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    import numpy as np

    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.array([], dtype=int)
    best_idx = selector.select(y, mu, rng)
    return np.array([best_idx])


def get_incumbent_index(
    selector,
    y: np.ndarray,
    rng: Generator,
    mu: np.ndarray | None = None,
) -> int:
    import numpy as np

    y = np.asarray(y, dtype=float)
    if y.size == 0:
        raise ValueError("y is empty")
    return int(selector.select(y, mu, rng))


def get_scalar_incumbent_value(
    selector,
    y_obs: np.ndarray,
    rng: Generator,
    *,
    mu_obs: np.ndarray | None = None,
) -> np.ndarray:
    import numpy as np

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


def generate_tr_candidates(
    compute_bounds_1d: Any,
    x_center: np.ndarray,
    lengthscales: np.ndarray | None,
    num_candidates: int,
    *,
    rng: Generator,
    candidate_rv: CandidateRV = CandidateRV.SOBOL,
    sobol_engine: QMCEngine | None = None,
) -> np.ndarray:
    from .turbo_utils import (
        generate_raasp_candidates,
        generate_raasp_candidates_uniform,
    )

    lb, ub = compute_bounds_1d(x_center, lengthscales)
    if candidate_rv == CandidateRV.SOBOL:
        if sobol_engine is None:
            raise ValueError(
                "sobol_engine is required when candidate_rv=CandidateRV.SOBOL"
            )
        return generate_raasp_candidates(
            x_center, lb, ub, num_candidates, rng=rng, sobol_engine=sobol_engine
        )
    if candidate_rv == CandidateRV.UNIFORM:
        return generate_raasp_candidates_uniform(
            x_center, lb, ub, num_candidates, rng=rng
        )
    raise ValueError(candidate_rv)
