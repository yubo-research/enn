from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .posterior_result import PosteriorResult
from .surrogate_result import SurrogateResult

if TYPE_CHECKING:
    from numpy.random import Generator


class NoSurrogate:
    def __init__(self) -> None:
        self._x_obs: np.ndarray | None = None
        self._y_obs: np.ndarray | None = None

    @property
    def lengthscales(self) -> np.ndarray | None:
        return getattr(self, "_lengthscales", None)

    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,
        rng: Generator | None = None,
    ) -> SurrogateResult:
        self._x_obs = np.asarray(x_obs, dtype=float)
        self._y_obs = np.asarray(y_obs, dtype=float)
        if self._y_obs.ndim == 1:
            self._y_obs = self._y_obs.reshape(-1, 1)
        return SurrogateResult(model=None, lengthscales=None)

    def predict(self, x: np.ndarray) -> PosteriorResult:
        if self._x_obs is None or self._y_obs is None:
            raise RuntimeError("NoSurrogate.predict requires fit() first")
        x = np.asarray(x, dtype=float)
        if np.array_equal(x, self._x_obs):
            return PosteriorResult(mu=self._y_obs.copy(), sigma=None)
        raise RuntimeError("NoSurrogate.predict only works for training points")

    def get_incumbent_candidate_indices(self, y_obs: np.ndarray) -> np.ndarray:
        return np.arange(len(y_obs), dtype=int)

    def sample(self, x: np.ndarray, num_samples: int, rng: Generator) -> np.ndarray:
        n = len(x)
        num_metrics = self._y_obs.shape[1] if hasattr(self, "_y_obs") else 1
        return rng.standard_normal((num_samples, n, num_metrics))
