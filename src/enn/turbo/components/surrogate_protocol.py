from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .surrogate_result import SurrogateResult
from .posterior_result import PosteriorResult

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

    from .incumbent_selector import IncumbentSelector


class Surrogate(Protocol):
    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,
        rng: Generator | None = None,
    ) -> SurrogateResult: ...

    def predict(self, x: np.ndarray) -> PosteriorResult: ...

    def sample(self, x: np.ndarray, num_samples: int, rng: Generator) -> np.ndarray: ...

    def find_x_center(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        selector: IncumbentSelector,
        rng: Generator,
    ) -> np.ndarray | None: ...
