from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine


@dataclass
class StaggerTrustRegion:
    num_dim: int
    num_arms: int
    low: float = 0.1
    high: float = 1.0

    def update(self, values: np.ndarray | Any) -> None:
        return

    def needs_restart(self) -> bool:
        return False

    def restart(self) -> None:
        return

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

    def generate_candidates(
        self,
        x_center: np.ndarray,
        weights: np.ndarray | None,
        num_candidates: int,
        rng: Generator,
        sobol_engine: QMCEngine,
    ) -> np.ndarray:
        import numpy as np

        from .turbo_utils import generate_raasp_candidates

        if num_candidates <= 0:
            raise ValueError(num_candidates)
        low = float(self.low)
        high = float(self.high)
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= 0.0:
            raise ValueError((low, high))
        if low > high:
            low = high
        log_min = float(np.log(low))
        log_max = float(np.log(high))
        length = float(np.exp(rng.uniform(log_min, log_max)))
        if weights is None:
            half_length = 0.5 * length
        else:
            half_length = weights * length / 2.0
        lb = np.clip(x_center - half_length, 0.0, 1.0)
        ub = np.clip(x_center + half_length, 0.0, 1.0)
        return generate_raasp_candidates(
            x_center, lb, ub, num_candidates, rng=rng, sobol_engine=sobol_engine
        )
