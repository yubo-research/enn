import numpy as np


class SimpleAdapter:
    def __init__(
        self,
        step_sizes: list[float],
        rng: np.random.Generator,
        p_success: float = 0.5,
        p_failure: float = 0.5,
    ) -> None:
        if len(step_sizes) == 0:
            raise ValueError("step_sizes must be non-empty")
        if not (0 <= p_success <= 1):
            raise ValueError("p_success must be in [0, 1]")
        if not (0 <= p_failure <= 1):
            raise ValueError("p_failure must be in [0, 1]")

        self._step_sizes = sorted(step_sizes)
        self._rng = rng
        self._p_success = p_success
        self._p_failure = p_failure
        self._idx = len(self._step_sizes) // 2  # Start in the middle

    def ask(self) -> float:
        return self._step_sizes[self._idx]

    def tell(self, success: bool) -> None:
        if success:
            if self._rng.random() < self._p_success:
                self._idx = len(self._step_sizes) - 1
        else:
            if self._idx > 0:
                if self._rng.random() < self._p_failure:
                    self._idx -= 1

    @property
    def step_size(self) -> float:
        return self._step_sizes[self._idx]
