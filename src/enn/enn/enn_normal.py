from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


@dataclass
class ENNNormal:
    mu: np.ndarray
    se: np.ndarray

    def sample(
        self,
        num_samples: int,
        rng: Generator,
        clip: float | None = None,
    ) -> np.ndarray:
        import numpy as np

        size = (*self.se.shape, num_samples)
        eps = rng.normal(size=size)
        if clip is not None:
            eps = np.clip(eps, a_min=-clip, a_max=clip)
        return np.expand_dims(self.mu, -1) + np.expand_dims(self.se, -1) * eps
