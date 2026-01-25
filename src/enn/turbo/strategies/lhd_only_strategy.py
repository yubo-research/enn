from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from ..sampling import draw_lhd
from .optimization_strategy import OptimizationStrategy

if TYPE_CHECKING:
    from ..optimizer import Optimizer
    from ..types import TellInputs


@dataclass
class LHDOnlyStrategy(OptimizationStrategy):
    _bounds: np.ndarray
    _rng: object

    @classmethod
    def create(cls, *, bounds: np.ndarray, rng: object) -> LHDOnlyStrategy:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must be (d, 2), got {bounds.shape}")
        return cls(_bounds=bounds, _rng=rng)

    def ask(self, opt: Optimizer, num_arms: int) -> np.ndarray:
        return draw_lhd(bounds=self._bounds, num_arms=num_arms, rng=opt._rng)

    def init_progress(self) -> tuple[int, int] | None:
        return None

    def tell(
        self, opt: Optimizer, inputs: TellInputs, *, x_unit: np.ndarray
    ) -> np.ndarray:
        del x_unit
        opt._y_tr_list = inputs.y.tolist()
        return np.asarray(inputs.y, dtype=float)
