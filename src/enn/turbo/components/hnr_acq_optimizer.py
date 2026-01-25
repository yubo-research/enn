from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from .thompson_acq_optimizer import ThompsonAcqOptimizer
from .ucb_acq_optimizer import UCBAcqOptimizer

if TYPE_CHECKING:
    from numpy.random import Generator
    from .protocols import Surrogate


class HnRAcqOptimizer:
    def __init__(
        self,
        base_optimizer: ThompsonAcqOptimizer | UCBAcqOptimizer,
        num_iterations: int = 100,
    ) -> None:
        self._base = base_optimizer
        self._num_iterations = num_iterations

    def _score_fn_ucb(
        self, x_pt: np.ndarray, surrogate: Surrogate, beta: float = 1.0
    ) -> float:
        posterior = surrogate.predict(x_pt.reshape(1, -1))
        mu = float(posterior.mu[0, 0])
        sigma = float(posterior.sigma[0, 0]) if posterior.sigma is not None else 0.0
        return mu + beta * sigma

    def _score_fn_thompson(
        self, x_pt: np.ndarray, surrogate: Surrogate, seed: int
    ) -> float:
        fixed_rng = np.random.default_rng(seed)
        samples = surrogate.sample(x_pt.reshape(1, -1), 1, fixed_rng)
        return float(samples[0, 0, 0])

    def _optimize_one_arm(
        self,
        x_start: np.ndarray,
        num_dim: int,
        rng: Generator,
        score_fn,
    ) -> np.ndarray:
        x_current = x_start.copy()
        current_score = score_fn(x_current)
        for _ in range(self._num_iterations):
            direction = rng.standard_normal(num_dim)
            direction = direction / np.linalg.norm(direction)
            step_size = rng.uniform(0.01, 0.1)
            x_proposed = np.clip(x_current + step_size * direction, 0.0, 1.0)
            proposed_score = score_fn(x_proposed)
            if proposed_score > current_score:
                x_current = x_proposed
                current_score = proposed_score
        return x_current

    def select(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        surrogate: Surrogate,
        rng: Generator,
        *,
        tr_state: Any | None = None,
    ) -> np.ndarray:
        num_dim = x_cand.shape[1]
        x_arms = np.zeros((num_arms, num_dim), dtype=float)
        is_ucb = isinstance(self._base, UCBAcqOptimizer)
        for arm_idx in range(num_arms):
            start_idx = rng.integers(0, len(x_cand))
            x_start = x_cand[start_idx]
            if is_ucb:
                beta = getattr(self._base, "_beta", 1.0)

                def score_fn(x_pt):
                    return self._score_fn_ucb(x_pt, surrogate, beta)
            else:
                seed = int(rng.integers(0, 2**31))

                def score_fn(x_pt, s=seed):
                    return self._score_fn_thompson(x_pt, surrogate, s)

            x_arms[arm_idx] = self._optimize_one_arm(x_start, num_dim, rng, score_fn)
        return x_arms
