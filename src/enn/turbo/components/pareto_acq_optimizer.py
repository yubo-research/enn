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
        tr_state: Any | None = None,
    ) -> np.ndarray:
        from enn.enn.enn_util import arms_from_pareto_fronts

        posterior = surrogate.predict(x_cand)
        mu = posterior.mu
        se = posterior.sigma if posterior.sigma is not None else np.zeros_like(mu)
        if mu.ndim == 2 and mu.shape[1] > 1:
            from nds import ndomsort

            n = mu.shape[0]
            i_keep: list[int] = []
            remaining_mask = np.ones(n, dtype=bool)
            while len(i_keep) < num_arms and np.any(remaining_mask):
                remaining_indices = np.where(remaining_mask)[0]
                fronts = ndomsort.non_domin_sort(
                    -mu[remaining_indices], only_front_indices=True
                )
                front_indices = remaining_indices[np.where(fronts == 0)[0]]
                if len(i_keep) + len(front_indices) <= num_arms:
                    i_keep.extend(front_indices.tolist())
                    remaining_mask[front_indices] = False
                else:
                    needed = num_arms - len(i_keep)
                    selected = rng.choice(front_indices, size=needed, replace=False)
                    i_keep.extend(selected.tolist())
                    break
            return x_cand[np.array(i_keep, dtype=int)]
        else:
            mu_1d = mu[:, 0] if mu.ndim == 2 else mu
            se_1d = se[:, 0] if se.ndim == 2 else se
            return arms_from_pareto_fronts(x_cand, mu_1d, se_1d, num_arms, rng)
