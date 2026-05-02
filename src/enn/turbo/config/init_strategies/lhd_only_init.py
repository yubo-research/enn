from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..init_strategy_base import InitStrategy

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


@dataclass(frozen=True)
class LHDOnlyInit(InitStrategy):
    def create_runtime_strategy(
        self,
        *,
        bounds: np.ndarray,
        rng: Generator,
        num_init: int | None,
    ) -> Any:
        from ...strategies.lhd_only_strategy import LHDOnlyStrategy

        del num_init
        return LHDOnlyStrategy.create(bounds=bounds, rng=rng)
