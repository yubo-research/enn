from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from ..trust_region import InitStrategy

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from ...strategies import OptimizationStrategy


@dataclass(frozen=True)
class LHDOnlyInit(InitStrategy):
    def create_runtime_strategy(
        self,
        *,
        bounds: np.ndarray,
        rng: Generator,
        num_init: int | None,
    ) -> OptimizationStrategy:
        from ...strategies import LHDOnlyStrategy

        del num_init
        return LHDOnlyStrategy.create(bounds=bounds, rng=rng)
