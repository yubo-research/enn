from __future__ import annotations
from typing import TYPE_CHECKING, Any, Protocol
from .posterior_result import PosteriorResult
from .surrogate_result import SurrogateResult

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


class Surrogate(Protocol):
    @property
    def lengthscales(self) -> np.ndarray | None: ...
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
        tr_state: Any,
        rng: Generator,
    ) -> np.ndarray | None: ...
    def get_incumbent_candidate_indices(self, y_obs: np.ndarray) -> np.ndarray: ...
