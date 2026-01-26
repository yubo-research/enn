from __future__ import annotations
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


class IncumbentSelector(Protocol):
    def select(
        self,
        y_obs: np.ndarray,
        mu_obs: np.ndarray | None,
        rng: Generator,
    ) -> int: ...
    def reset(self, rng: Generator) -> None: ...
