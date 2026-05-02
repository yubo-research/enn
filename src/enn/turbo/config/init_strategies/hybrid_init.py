from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..init_strategy_base import InitStrategy

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


@dataclass(frozen=True)
class HybridInit(InitStrategy):
    def create_runtime_strategy(
        self,
        *,
        bounds: np.ndarray,
        rng: Generator,
        num_init: int | None,
    ) -> Any:
        from ...strategies.turbo_hybrid_strategy import TurboHybridStrategy

        return TurboHybridStrategy.create(bounds=bounds, rng=rng, num_init=num_init)
