import numpy as np
from torch import nn

from uhd.perturb_module import perturb_module, unperturb_module


class SimpleRacer:
    def __init__(self, step_size: float, rng: np.random.Generator) -> None:
        self._step_size = step_size
        self._rng = rng
        self._incumbent_y: float = float("-inf")
        self._challenger_seed: int | None = None
        self._accepts: int = 0
        self._races: int = 0

    def ask(self, module: nn.Module) -> int:
        self._challenger_seed = int(self._rng.integers(1, 2**31))
        perturb_module(module, self._challenger_seed, self._step_size)
        return self._challenger_seed

    def tell(self, module: nn.Module, seed: int, y: float) -> None:
        assert np.isfinite(y)
        assert seed == self._challenger_seed

        self._races += 1
        if y > self._incumbent_y:
            self._incumbent_y = y
            self._accepts += 1
        else:
            unperturb_module(module, self._challenger_seed, self._step_size)

    @property
    def accepts(self) -> int:
        return self._accepts

    @property
    def races(self) -> int:
        return self._races

    @property
    def incumbent_y(self) -> float:
        return self._incumbent_y
