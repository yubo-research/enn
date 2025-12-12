from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

from .base_turbo_impl import BaseTurboImpl


class LHDOnlyImpl(BaseTurboImpl):
    def get_x_center(
        self,
        x_obs_list: list,
        y_obs_list: list,
        rng: Generator,
    ) -> np.ndarray | None:
        return None

    def select_candidates(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        num_dim: int,
        rng: Generator,
        fallback_fn: Callable[[np.ndarray, int], np.ndarray],
        from_unit_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        from .turbo_utils import latin_hypercube

        unit = latin_hypercube(num_arms, num_dim, rng=rng)
        return from_unit_fn(unit)

    def update_trust_region(
        self,
        tr_state: Any,
        y_obs_list: list,
        x_center: np.ndarray | None = None,
        k: int | None = None,
    ) -> None:
        pass
