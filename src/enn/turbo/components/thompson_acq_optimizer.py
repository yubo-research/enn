from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator
    from .protocols import Surrogate


class ThompsonAcqOptimizer:
    def select(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        surrogate: Surrogate,
        rng: Generator,
        *,
        tr_state: Any | None = None,
    ) -> np.ndarray:
        from ..turbo_utils import argmax_random_tie

        num_candidates = len(x_cand)
        samples = surrogate.sample(x_cand, num_arms, rng)
        assert samples.ndim == 3, f"samples.ndim={samples.ndim}, expected 3"
        assert samples.shape[0] == num_arms, (
            f"samples.shape[0]={samples.shape[0]}, expected num_arms={num_arms}"
        )
        assert samples.shape[1] == num_candidates, (
            f"samples.shape[1]={samples.shape[1]}, expected num_candidates={num_candidates}"
        )
        num_metrics = samples.shape[2]
        if tr_state is not None and hasattr(tr_state, "scalarize"):
            indices = []
            # Vectorize scalarization across all arms and candidates
            # samples is (num_arms, num_candidates, num_metrics)
            # Reshape to (num_arms * num_candidates, num_metrics) for scalarize
            flat_samples = samples.reshape(-1, num_metrics)
            flat_scores = tr_state.scalarize(flat_samples, clip=False)
            all_scores = flat_scores.reshape(num_arms, num_candidates)

            for i in range(num_arms):
                scores = all_scores[i].copy()
                for prev_idx in indices:
                    scores[prev_idx] = -np.inf
                idx = argmax_random_tie(scores, rng=rng)
                indices.append(idx)
            return x_cand[indices]
        else:
            arm_indices = [
                argmax_random_tie(samples[i, :, 0], rng=rng) for i in range(num_arms)
            ]
            return x_cand[arm_indices]
