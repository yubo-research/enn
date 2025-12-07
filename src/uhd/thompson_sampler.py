from typing import Any

import numpy as np

from uhd.mvue_accumulator import MVUE


class ThompsonSampler:
    def __init__(self, arms: list[Any], rng: np.random.Generator, decay: float) -> None:
        if len(arms) == 0:
            raise ValueError("arms must be non-empty")
        self._arms = list(arms)
        self._rng = rng
        self._mvues = [MVUE(decay=decay) for _ in self._arms]

    def ask(self) -> Any:
        uninitialized = [i for i, m in enumerate(self._mvues) if m.n < 3]
        if uninitialized:
            idx = self._rng.choice(uninitialized)
            return self._arms[idx]

        samples = [self._rng.normal(m.mean, m.se) for m in self._mvues]
        idx = int(np.argmax(samples))
        return self._arms[idx]

    def tell(self, arm: Any, y: float, y_var: float) -> None:
        idx = self._find_arm_index(arm)
        self._mvues[idx].update(y, y_var)

    def _find_arm_index(self, arm: Any) -> int:
        for i, a in enumerate(self._arms):
            if a is arm:
                return i
        raise ValueError("arm not found")
