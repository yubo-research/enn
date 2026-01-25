from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..optimizer import Optimizer
    from ..types import TellInputs


class OptimizationStrategy(ABC):
    @abstractmethod
    def ask(self, opt: Optimizer, num_arms: int) -> np.ndarray: ...
    @abstractmethod
    def tell(
        self, opt: Optimizer, inputs: TellInputs, *, x_unit: np.ndarray
    ) -> np.ndarray: ...
    @abstractmethod
    def init_progress(self) -> tuple[int, int] | None: ...
