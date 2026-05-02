from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..types import TellInputs


class OptimizationStrategy(ABC):
    @abstractmethod
    def ask(self, opt: Any, num_arms: int) -> np.ndarray: ...
    @abstractmethod
    def tell(
        self, opt: Any, inputs: TellInputs, *, x_unit: np.ndarray
    ) -> np.ndarray: ...
    @abstractmethod
    def init_progress(self) -> tuple[int, int] | None: ...
