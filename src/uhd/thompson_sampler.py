from typing import Any

import numpy as np

from uhd.mvue_accumulator import MVUE


class ThompsonSampler:
    def __init__(
        self,
        arms: list[Any],
        rng: np.random.Generator,
        decay: float = 1.0,
        min_observations: int = 3,
    ) -> None:
        if len(arms) == 0:
            raise ValueError("arms must be non-empty")
        if min_observations < 1:
            raise ValueError("min_observations must be at least 1")
        self._arms = list(arms)
        self._rng = rng
        self._min_observations = min_observations
        self._mvues = [MVUE(decay=decay) for _ in self._arms]
        self._last_idx: int | None = None

    def ask(self) -> Any:
        uninitialized = [
            i for i, m in enumerate(self._mvues) if m.n < self._min_observations
        ]
        if uninitialized:
            self._last_idx = self._rng.choice(uninitialized)
        else:
            samples = [self._rng.normal(m.mean, m.se) for m in self._mvues]
            self._last_idx = int(np.argmax(samples))
        return self._arms[self._last_idx]

    def tell(self, success: bool) -> None:
        if self._last_idx is None:
            raise RuntimeError("tell() called before ask()")
        reward = 1 if success else 0
        self._mvues[self._last_idx].update(reward, 1)
