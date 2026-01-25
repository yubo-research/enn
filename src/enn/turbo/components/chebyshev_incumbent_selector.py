from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


@dataclass
class ChebyshevIncumbentSelector:
    num_metrics: int
    noise_aware: bool
    alpha: float
    _weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.num_metrics < 1:
            raise ValueError(f"num_metrics must be >= 1, got {self.num_metrics}")

    @property
    def weights(self) -> np.ndarray | None:
        return self._weights

    def _sample_weights(self, rng: Generator) -> None:
        import numpy as np

        alpha = np.ones(self.num_metrics, dtype=float)
        self._weights = np.asarray(rng.dirichlet(alpha), dtype=float)

    def reset(self, rng: Generator) -> None:
        self._sample_weights(rng)

    def select(
        self,
        y_obs: np.ndarray,
        mu_obs: np.ndarray | None,
        rng: Generator,
    ) -> int:
        import numpy as np
        from ..turbo_utils import argmax_random_tie

        if self._weights is None:
            self._sample_weights(rng)
        y = np.asarray(y_obs, dtype=float)
        if y.ndim != 2 or y.shape[1] != self.num_metrics:
            raise ValueError(
                f"Expected y with {self.num_metrics} metrics, got {y.shape}"
            )
        if self.noise_aware:
            if mu_obs is None:
                raise ValueError(
                    "noise_aware=True requires a surrogate that provides mu. "
                    "Either use a GP/ENN surrogate or set noise_aware=False."
                )
            values = np.asarray(mu_obs, dtype=float)
        else:
            values = y
        scores = self._scalarize(values)
        return int(argmax_random_tie(scores, rng=rng))

    def _scalarize(self, values: np.ndarray) -> np.ndarray:
        import numpy as np

        if self._weights is None:
            raise RuntimeError("Weights not initialized; call reset() first")
        v_min = values.min(axis=0)
        v_max = values.max(axis=0)
        denom = v_max - v_min
        is_deg = denom <= 0.0
        denom_safe = np.where(is_deg, 1.0, denom)
        z = (values - v_min.reshape(1, -1)) / denom_safe.reshape(1, -1)
        z = np.where(is_deg, 0.5, z)
        z = np.clip(z, 0.0, 1.0)
        t = z * self._weights.reshape(1, -1)
        return np.min(t, axis=1) + self.alpha * np.sum(t, axis=1)
