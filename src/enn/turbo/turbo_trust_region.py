from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

    from .components.incumbent_selector import IncumbentSelector
    from .config.turbo_tr_config import TurboTRConfig


@dataclass
class TurboTrustRegion:
    config: TurboTRConfig
    num_dim: int
    length: float = field(init=False)
    failure_counter: int = 0
    success_counter: int = 0
    best_value: float = -float("inf")
    prev_num_obs: int = 0
    incumbent_selector: IncumbentSelector | None = field(default=None, repr=False)
    _num_arms: int | None = field(default=None, repr=False)
    _failure_tolerance: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from .components.incumbent_selector import ScalarIncumbentSelector

        self.length = self.config.length_init
        self.success_tolerance = 3
        if self.incumbent_selector is None:
            self.incumbent_selector = ScalarIncumbentSelector(noise_aware=True)

    @property
    def length_init(self) -> float:
        return self.config.length_init

    @property
    def length_min(self) -> float:
        return self.config.length_min

    @property
    def length_max(self) -> float:
        return self.config.length_max

    @property
    def num_metrics(self) -> int:
        """Single-objective trust region always has 1 metric."""
        return 1

    def _ensure_initialized(self, num_arms: int) -> None:
        import numpy as np

        if self._num_arms is None:
            self._num_arms = num_arms
            self._failure_tolerance = int(
                np.ceil(
                    max(
                        4.0 / float(num_arms),
                        float(self.num_dim) / float(num_arms),
                    )
                )
            )
        elif num_arms != self._num_arms:
            raise ValueError(
                f"num_arms changed from {self._num_arms} to {num_arms}; "
                "must be consistent across ask() calls"
            )

    @property
    def failure_tolerance(self) -> int:
        if self._failure_tolerance is None:
            raise RuntimeError("failure_tolerance not initialized; call ask() first")
        return self._failure_tolerance

    def update(self, values: np.ndarray | Any) -> None:
        import numpy as np

        # Skip counter updates until first ask() initializes failure_tolerance
        if self._failure_tolerance is None:
            return

        values = np.asarray(values, dtype=float)
        if values.ndim == 2:
            if values.shape[1] != 1:
                raise ValueError(f"TurboTrustRegion expects m=1, got {values.shape}")
            values = values[:, 0]
        elif values.ndim != 1:
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
        # Use a shift-invariant scale for the improvement tolerance so that
        # trust-region behavior is invariant to affine transforms of y.
        prev_values = values[: self.prev_num_obs]
        scale = (
            float(np.max(prev_values) - np.min(prev_values))
            if prev_values.size
            else 0.0
        )
        improved = np.max(new_values) > self.best_value + 1e-3 * scale
        if improved:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1
        if self.success_counter >= self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter >= self._failure_tolerance:
            self.length = 0.5 * self.length
            self.failure_counter = 0

        self.best_value = max(self.best_value, float(np.max(new_values)))
        self.prev_num_obs = values.size

    def needs_restart(self) -> bool:
        return self.length < self.length_min

    def restart(self, rng: Any | None = None) -> None:  # noqa: ARG002
        self.length = self.length_init
        self.failure_counter = 0
        self.success_counter = 0
        self.best_value = -float("inf")
        self.prev_num_obs = 0

    def validate_request(self, num_arms: int, *, is_fallback: bool = False) -> None:  # noqa: ARG002
        self._ensure_initialized(num_arms)

    def compute_bounds_1d(
        self, x_center: np.ndarray | Any, lengthscales: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        if lengthscales is None:
            half_length = 0.5 * self.length
        else:
            lengthscales = np.asarray(lengthscales, dtype=float).reshape(-1)
            if lengthscales.shape != (self.num_dim,):
                raise ValueError(
                    f"lengthscales must have shape ({self.num_dim},), got {lengthscales.shape}"
                )
            if not np.all(np.isfinite(lengthscales)):
                raise ValueError("lengthscales must be finite")
            half_length = lengthscales * self.length / 2.0
        lb = np.clip(x_center - half_length, 0.0, 1.0)
        ub = np.clip(x_center + half_length, 0.0, 1.0)
        return lb, ub

    def get_incumbent_indices(
        self,
        y: np.ndarray | Any,
        rng: Generator,
        mu: np.ndarray | None = None,
    ) -> np.ndarray:
        from .tr_helpers import get_single_incumbent_index

        return get_single_incumbent_index(self.incumbent_selector, y, rng, mu)
