from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

from .base_turbo_impl import BaseTurboImpl


class TurboZeroImpl(BaseTurboImpl):
    def select_candidates(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        num_dim: int,
        rng: Generator,
        fallback_fn: Callable[[np.ndarray, int], np.ndarray],
        from_unit_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        from .proposal import select_uniform

        return select_uniform(x_cand, num_arms, num_dim, rng, from_unit_fn)
