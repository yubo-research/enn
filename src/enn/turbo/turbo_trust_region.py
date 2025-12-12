from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine


@dataclass
class TurboTrustRegion:
    num_dim: int
    num_arms: int
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    best_value: float = -float("inf")
    prev_num_obs: int = 0

    def __post_init__(self) -> None:
        import numpy as np

        self.failure_tolerance = int(
            np.ceil(
                max(
                    4.0 / float(self.num_arms),
                    float(self.num_dim) / float(self.num_arms),
                )
            )
        )
        self.success_tolerance = 3

    def update(self, values: np.ndarray | Any) -> None:
        import numpy as np

        if values.ndim != 1:
            raise ValueError(values.shape)
        if values.size == 0:
            return
        new_values = values[self.prev_num_obs :]
        if new_values.size == 0:
            return
        if not np.isfinite(self.best_value):
            self.best_value = float(np.max(new_values))
            self.prev_num_obs = values.size
            return
        improved = np.max(new_values) > self.best_value + 1e-3 * np.abs(self.best_value)
        if improved:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1
        if self.success_counter >= self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter >= self.failure_tolerance:
            self.length = 0.5 * self.length
            self.failure_counter = 0

        self.best_value = max(self.best_value, float(np.max(new_values)))
        self.prev_num_obs = values.size

    def needs_restart(self) -> bool:
        return self.length < self.length_min

    def restart(self) -> None:
        self.length = self.length_init
        self.failure_counter = 0
        self.success_counter = 0
        self.best_value = -float("inf")
        self.prev_num_obs = 0

    def validate_request(self, num_arms: int, *, is_fallback: bool = False) -> None:
        if is_fallback:
            if num_arms > self.num_arms:
                raise ValueError(
                    f"num_arms {num_arms} > configured num_arms {self.num_arms}"
                )
        else:
            if num_arms != self.num_arms:
                raise ValueError(
                    f"num_arms {num_arms} != configured num_arms {self.num_arms}"
                )

    def compute_bounds_1d(
        self, x_center: np.ndarray | Any, weights: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        if weights is None:
            half_length = 0.5 * self.length
        else:
            half_length = weights * self.length / 2.0
        lb = np.clip(x_center - half_length, 0.0, 1.0)
        ub = np.clip(x_center + half_length, 0.0, 1.0)
        return lb, ub

    def generate_candidates(
        self,
        x_center: np.ndarray,
        weights: np.ndarray | None,
        num_candidates: int,
        rng: Generator,
        sobol_engine: QMCEngine,
    ) -> np.ndarray:
        from .turbo_utils import raasp

        lb, ub = self.compute_bounds_1d(x_center, weights)
        return raasp(
            x_center,
            lb,
            ub,
            num_candidates,
            num_pert=20,
            rng=rng,
            sobol_engine=sobol_engine,
        )
