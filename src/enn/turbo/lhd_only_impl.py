from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

from .base_turbo_impl import BaseTurboImpl
from .turbo_config import LHDOnlyConfig


class LHDOnlyImpl(BaseTurboImpl):
    def __init__(self, config: LHDOnlyConfig) -> None:
        super().__init__(config)

    def get_x_center(
        self,
        x_obs_list: list,
        y_obs_list: list,
        rng: Generator,
        tr_state: Any = None,
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
        tr_state: Any = None,  # noqa: ARG002
    ) -> np.ndarray:
        from .turbo_utils import latin_hypercube

        unit = latin_hypercube(num_arms, num_dim, rng=rng)
        return from_unit_fn(unit)

    def update_trust_region(
        self,
        tr_state: Any,
        x_obs_list: list,
        y_obs_list: list,
        x_center: np.ndarray | None = None,
        k: int | None = None,
    ) -> None:
        pass
