from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine


class TrustRegionCandidateMixin:
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
