from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from torch import nn

from uhd.perturb_module import perturb_module, unperturb_module

if TYPE_CHECKING:
    from uhd.simple_adapter import SimpleAdapter
    from uhd.thompson_sampler import ThompsonSampler


class SimpleRacer:
    def __init__(
        self,
        adapter: ThompsonSampler | SimpleAdapter,
        rng: np.random.Generator,
        momentum: bool = False,
    ) -> None:
        self._adapter = adapter
        self._rng = rng
        self._momentum = momentum

        self._incumbent_y: float = float("-inf")
        self._incumbent_y_var: float = float("inf")
        self._challenger_seed: int | None = None
        self._challenger_step_size: float | None = None
        self._accepts: int = 0
        self._races: int = 0
        self._last_accepted: bool = False

    def ask(self, module: nn.Module) -> int:
        self._challenger_step_size = self._adapter.ask()

        # If momentum enabled and last perturbation was accepted, reuse the same seed
        if (
            not (self._momentum and self._last_accepted)
            or self._challenger_seed is None
        ):
            self._challenger_seed = int(self._rng.integers(1, 2**31))

        perturb_module(module, self._challenger_seed, self._challenger_step_size)
        return self._challenger_seed

    def tell(self, module: nn.Module, seed: int, y: float, y_var: float) -> bool:
        if not np.isfinite(y):
            raise ValueError("y must be finite")
        if not np.isfinite(y_var) or y_var <= 0:
            raise ValueError("y_var must be finite and positive")
        if seed != self._challenger_seed:
            raise ValueError("seed mismatch")

        accepted = y > self._incumbent_y

        # Update the adapter
        if np.isfinite(self._incumbent_y):
            self._adapter.tell(accepted)

        self._races += 1
        if accepted:
            self._incumbent_y = y
            self._incumbent_y_var = y_var
            self._accepts += 1
            self._last_accepted = True
            return True
        else:
            unperturb_module(module, self._challenger_seed, self._challenger_step_size)
            self._last_accepted = False
            return False

    @property
    def accepts(self) -> int:
        return self._accepts

    @property
    def races(self) -> int:
        return self._races

    @property
    def incumbent_y(self) -> float:
        return self._incumbent_y

    @property
    def step_size(self) -> float | None:
        return self._challenger_step_size
