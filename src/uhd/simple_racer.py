import numpy as np
from torch import nn

from uhd.perturb_module import perturb_module, unperturb_module


class SimpleRacer:
    def __init__(
        self, step_sizes: list[float], rng: np.random.Generator, decay: float = None
    ) -> None:
        if len(step_sizes) == 0:
            raise ValueError("step_sizes must be non-empty")
        self._step_sizes = list(step_sizes)
        self._rng = rng
        self._decay = decay
        self._incumbent_y: float = float("-inf")
        self._incumbent_y_var: float = float("inf")
        self._challenger_seed: int | None = None
        self._challenger_step_size: float | None = None
        self._accepts: int = 0
        self._races: int = 0

        if len(self._step_sizes) > 1:
            from uhd.thompson_sampler import ThompsonSampler

            self._sampler: ThompsonSampler | None = ThompsonSampler(
                self._step_sizes, rng, decay=decay
            )
        else:
            assert decay is None, "decay must be None if there is only one step size"
            self._sampler = None

    def ask(self, module: nn.Module) -> int:
        if self._sampler is not None:
            self._challenger_step_size = self._sampler.ask()
        else:
            self._challenger_step_size = self._step_sizes[0]

        self._challenger_seed = int(self._rng.integers(1, 2**31))
        perturb_module(module, self._challenger_seed, self._challenger_step_size)
        return self._challenger_seed

    def tell(self, module: nn.Module, seed: int, y: float, y_var: float) -> None:
        if not np.isfinite(y):
            raise ValueError("y must be finite")
        if not np.isfinite(y_var) or y_var <= 0:
            raise ValueError("y_var must be finite and positive")
        if seed != self._challenger_seed:
            raise ValueError("seed mismatch")

        if self._sampler is not None and np.isfinite(self._incumbent_y):
            improvement = max(0.0, y - self._incumbent_y)
            improvement_var = y_var + self._incumbent_y_var
            self._sampler.tell(self._challenger_step_size, improvement, improvement_var)

        self._races += 1
        if y > self._incumbent_y:
            self._incumbent_y = y
            self._incumbent_y_var = y_var
            self._accepts += 1
            return True
        else:
            unperturb_module(module, self._challenger_seed, self._challenger_step_size)
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
