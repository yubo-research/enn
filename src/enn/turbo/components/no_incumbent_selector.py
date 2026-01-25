from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


@dataclass
class NoIncumbentSelector:
    def select(
        self,
        y_obs: np.ndarray,
        mu_obs: np.ndarray | None,
        rng: Generator,
    ) -> int:
        return 0

    def reset(self, rng: Generator) -> None:
        pass
