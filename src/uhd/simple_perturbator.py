from __future__ import annotations

import numpy as np
from torch import nn

from uhd.perturb_module import perturb_module, unperturb_module


class SimplePerturbator:
    def __init__(
        self,
        rng: np.random.Generator,
        momentum: bool = False,
    ) -> None:
        self._rng = rng
        self._momentum = momentum
        self._seed: int | None = None
        self._step_size: float | None = None
        self._last_accepted: bool = False
        self._incumbent_y: float = float("-inf")

    def ask(self, module: nn.Module, step_size: float) -> int:
        if not (self._momentum and self._last_accepted) or self._seed is None:
            self._seed = int(self._rng.integers(1, 2**31))

        self._step_size = step_size
        perturb_module(module, self._seed, step_size)
        return self._seed

    def tell(self, module: nn.Module, seed: int, y: float, y_var: float) -> None:
        accepted = y > self._incumbent_y
        if accepted:
            self._incumbent_y = y
            self._last_accepted = True
        else:
            unperturb_module(module, self._seed, self._step_size)
            self._last_accepted = False
