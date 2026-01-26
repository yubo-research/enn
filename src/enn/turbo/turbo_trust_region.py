from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from .turbo_utils import ScalarIncumbentMixin

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from .components.incumbent_selector import IncumbentSelector
    from .config.turbo_tr_config import TurboTRConfig


@dataclass
class TurboTrustRegion(ScalarIncumbentMixin):
    config: TurboTRConfig
    num_dim: int
    length: float = field(init=False)
    failure_counter: int = 0
    success_counter: int = 0
    best_value: float = -float("inf")
    prev_num_obs: int = 0
    incumbent_selector: IncumbentSelector = field(default=None, repr=False)
    _num_arms: int | None = field(default=None, repr=False)
    _failure_tolerance: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from .components.incumbent_selector import ScalarIncumbentSelector

        self.length = self.config.length_init
        self.success_tolerance = 3
        if self.incumbent_selector is None:
            self.incumbent_selector = ScalarIncumbentSelector(noise_aware=False)

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
        return 1

    def _ensure_initialized(self, num_arms: int) -> None:
        import numpy as np

        if self._num_arms is None:
            self._num_arms, self._failure_tolerance = (
                num_arms,
                int(
                    np.ceil(
                        max(
                            4.0 / float(num_arms), float(self.num_dim) / float(num_arms)
                        )
                    )
                ),
            )
        elif num_arms != self._num_arms:
            raise ValueError(
                f"num_arms changed from {self._num_arms} to {num_arms}; must be consistent across ask() calls"
            )
        assert self._failure_tolerance is not None

    @property
    def failure_tolerance(self) -> int:
        if self._failure_tolerance is None:
            raise RuntimeError("failure_tolerance not initialized; call ask() first")
        return self._failure_tolerance

    def _coerce_y_obs_1d(self, y_obs: np.ndarray | Any) -> np.ndarray:
        import numpy as np

        y_obs = np.asarray(y_obs, dtype=float)
        if y_obs.ndim == 2:
            if y_obs.shape[1] != 1:
                raise ValueError(f"TurboTrustRegion expects m=1, got {y_obs.shape}")
            return y_obs[:, 0]
        if y_obs.ndim != 1:
            raise ValueError(y_obs.shape)
        return y_obs

    def _coerce_y_incumbent_value(self, y_incumbent: np.ndarray | Any) -> float:
        import numpy as np

        y_incumbent = np.asarray(y_incumbent, dtype=float).reshape(-1)
        if y_incumbent.shape != (self.num_metrics,):
            raise ValueError(
                f"y_incumbent must have shape ({self.num_metrics},), got {y_incumbent.shape}"
            )
        return float(y_incumbent[0])

    def _improvement_scale(self, prev_values: np.ndarray) -> float:
        import numpy as np

        if prev_values.size == 0:
            return 0.0
        return float(np.max(prev_values) - np.min(prev_values))

    def _update_counters_and_length(self, *, improved: bool) -> None:
        self.success_counter, self.failure_counter = (
            (self.success_counter + 1, 0) if improved else (0, self.failure_counter + 1)
        )
        if self.success_counter >= self.success_tolerance:
            self.length, self.success_counter = (
                min(2.0 * self.length, self.length_max),
                0,
            )
        elif (
            self._failure_tolerance is not None
            and self.failure_counter >= self._failure_tolerance
        ):
            self.length, self.failure_counter = 0.5 * self.length, 0

    def update(self, y_obs: np.ndarray | Any, y_incumbent: np.ndarray | Any) -> None:
        if self._failure_tolerance is None:
            return
        y_obs = self._coerce_y_obs_1d(y_obs)
        n = int(y_obs.size)
        if n <= 0 or n == self.prev_num_obs:
            return
        if n < self.prev_num_obs:
            raise ValueError((n, self.prev_num_obs))
        y_incumbent_value = self._coerce_y_incumbent_value(y_incumbent)
        import math

        if not math.isfinite(self.best_value):
            self.best_value, self.prev_num_obs = y_incumbent_value, n
            return
        prev_values = y_obs[: self.prev_num_obs]
        scale = self._improvement_scale(prev_values)
        self._update_counters_and_length(
            improved=y_incumbent_value > self.best_value + 1e-3 * scale
        )
        self.best_value, self.prev_num_obs = max(self.best_value, y_incumbent_value), n

    def needs_restart(self) -> bool:
        return self.length < self.length_min

    def restart(self, rng: Any | None = None) -> None:
        (
            self.length,
            self.failure_counter,
            self.success_counter,
            self.best_value,
            self.prev_num_obs,
            self._num_arms,
            self._failure_tolerance,
        ) = self.length_init, 0, 0, -float("inf"), 0, None, None

    def validate_request(self, num_arms: int, *, is_fallback: bool = False) -> None:
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
        import numpy as np

        return np.array([self.get_incumbent_index(y, rng, mu=mu)])
