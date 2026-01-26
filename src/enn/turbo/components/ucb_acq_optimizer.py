from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator
    from .protocols import Surrogate


class UCBAcqOptimizer:
    def __init__(self, beta: float = 1.0) -> None:
        self._beta = beta

    def select(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        surrogate: Surrogate,
        rng: Generator,
        *,
        tr_state: Any | None = None,
    ) -> np.ndarray:
        num_candidates = len(x_cand)
        posterior = surrogate.predict(x_cand)
        mu = posterior.mu
        sigma = posterior.sigma if posterior.sigma is not None else np.zeros_like(mu)
        assert mu.ndim == 2, f"mu.ndim={mu.ndim}, expected 2"
        assert mu.shape[0] == num_candidates, (
            f"mu.shape[0]={mu.shape[0]}, expected {num_candidates}"
        )
        num_metrics = mu.shape[1]
        if tr_state is not None and hasattr(tr_state, "scalarize"):
            ucb = mu + self._beta * sigma
            assert ucb.shape == (
                num_candidates,
                num_metrics,
            ), f"ucb.shape={ucb.shape}, expected ({num_candidates}, {num_metrics})"
            scores = tr_state.scalarize(ucb, clip=False)
            assert scores.shape == (num_candidates,), (
                f"scores.shape={scores.shape}, expected ({num_candidates},)"
            )
        else:
            scores = mu[:, 0] + self._beta * sigma[:, 0]
            assert scores.shape == (num_candidates,)
        shuffled_indices = rng.permutation(len(scores))
        shuffled_scores = scores[shuffled_indices]
        top_k_in_shuffled = np.argpartition(-shuffled_scores, num_arms - 1)[:num_arms]
        idx = shuffled_indices[top_k_in_shuffled]
        return x_cand[idx]
