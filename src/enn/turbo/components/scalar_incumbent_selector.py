from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclass
class ScalarIncumbentSelector:
    noise_aware: bool

    def select(
        self,
        y_obs: np.ndarray,
        mu_obs: np.ndarray | None,
        rng: Generator,
    ) -> int:
        from ..turbo_utils import argmax_random_tie

        y = np.asarray(y_obs, dtype=float)
        if y.ndim == 2:
            y = y[:, 0]
        if self.noise_aware:
            if mu_obs is None:
                raise ValueError(
                    "noise_aware=True requires a surrogate that provides mu. "
                    "Either use a GP/ENN surrogate or set noise_aware=False."
                )
            mu = np.asarray(mu_obs, dtype=float)
            if mu.ndim == 2:
                mu = mu[:, 0]
            return int(argmax_random_tie(mu, rng=rng))
        return int(argmax_random_tie(y, rng=rng))

    def reset(self, rng: Generator) -> None:
        pass
