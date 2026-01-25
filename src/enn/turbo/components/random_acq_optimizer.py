from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator
    from .protocols import Surrogate


class RandomAcqOptimizer:
    def select(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        surrogate: Surrogate,
        rng: Generator,
        *,
        tr_state: Any | None = None,
    ) -> np.ndarray:
        idx = rng.choice(len(x_cand), size=num_arms, replace=False)
        return x_cand[idx]
