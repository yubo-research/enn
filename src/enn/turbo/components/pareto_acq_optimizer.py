from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from .protocols import Surrogate


class ParetoAcqOptimizer:
    def select(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        surrogate: Surrogate,
        rng: Generator,
        *,
        tr_state: Any | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        from enn.enn.enn_util import arms_from_pareto_fronts

        posterior = surrogate.predict(x_cand)
        mu = posterior.mu[:, 0]
        se = posterior.sigma[:, 0] if posterior.sigma is not None else np.zeros_like(mu)

        return arms_from_pareto_fronts(x_cand, mu, se, num_arms, rng)
