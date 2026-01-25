from __future__ import annotations
from typing import TYPE_CHECKING, Any, Protocol
from .surrogate_protocol import Surrogate

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


class AcquisitionOptimizer(Protocol):
    def select(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        surrogate: Surrogate,
        rng: Generator,
        *,
        tr_state: Any | None = None,
    ) -> np.ndarray: ...
