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


class TrustRegion(Protocol):
    @property
    def length(self) -> float: ...
    def compute_bounds(
        self, x_center: np.ndarray, lengthscales: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def update(self, y_obs: np.ndarray, y_incumbent: np.ndarray) -> None: ...
    def needs_restart(self) -> bool: ...
    def restart(self) -> None: ...
    def get_incumbent_indices(self, y: np.ndarray, rng: Generator) -> np.ndarray: ...
    def get_incumbent_index(
        self, y: np.ndarray, rng: Generator, mu: np.ndarray | None = None
    ) -> int: ...
    def get_incumbent_value(
        self, y_obs: np.ndarray, rng: Generator, mu_obs: np.ndarray | None = None
    ) -> np.ndarray: ...
