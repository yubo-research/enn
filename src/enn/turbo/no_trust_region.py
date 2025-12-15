from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine


@dataclass
class NoTrustRegion:
    num_dim: int
    num_arms: int
    length: float = 1.0

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

    def compute_bounds_1d(
        self, x_center: np.ndarray | Any, lengthscales: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        lb = np.zeros_like(x_center, dtype=float)
        ub = np.ones_like(x_center, dtype=float)
        return lb, ub

    def generate_candidates(
        self,
        x_center: np.ndarray,
        lengthscales: np.ndarray | None,
        num_candidates: int,
        rng: Generator,
        sobol_engine: QMCEngine,
    ) -> np.ndarray:
        from .turbo_utils import generate_trust_region_candidates

        return generate_trust_region_candidates(
            x_center,
            lengthscales,
            num_candidates,
            compute_bounds_1d=self.compute_bounds_1d,
            rng=rng,
            sobol_engine=sobol_engine,
        )
