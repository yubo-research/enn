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
        del bounds, rng, num_init
        raise RuntimeError(
            "LHDOnlyInit is a Rust-routing config marker for lhd_only_config; "
            "use create_optimizer, not python_fallback runtime strategies"
        )
