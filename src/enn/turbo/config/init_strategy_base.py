from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


class InitStrategy(ABC):
    @abstractmethod
    def create_runtime_strategy(
        self,
        *,
        bounds: np.ndarray,
        rng: Generator,
        num_init: int | None,
    ) -> Any:
        """Create the runtime strategy. Returns an OptimizationStrategy."""
        ...
