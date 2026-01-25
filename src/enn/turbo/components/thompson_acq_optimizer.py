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
        num_candidates = len(x_cand)
        samples = surrogate.sample(x_cand, num_arms, rng)
        assert samples.ndim == 3, f"samples.ndim={samples.ndim}, expected 3"
        assert (
            samples.shape[0] == num_arms
        ), f"samples.shape[0]={samples.shape[0]}, expected num_arms={num_arms}"
        assert (
            samples.shape[1] == num_candidates
        ), f"samples.shape[1]={samples.shape[1]}, expected num_candidates={num_candidates}"
        num_metrics = samples.shape[2]
        if tr_state is not None and hasattr(tr_state, "scalarize"):
            indices = []
            for i in range(num_arms):
                sample_i = samples[i]
                assert sample_i.shape == (num_candidates, num_metrics), (
                    f"sample_i.shape={sample_i.shape}, "
                    f"expected ({num_candidates}, {num_metrics})"
                )
                scores = tr_state.scalarize(sample_i, clip=False)
                assert scores.shape == (
                    num_candidates,
                ), f"scores.shape={scores.shape}, expected ({num_candidates},)"
                for prev_idx in indices:
                    scores[prev_idx] = -np.inf
                idx = np.argmax(scores)
                indices.append(idx)
            return x_cand[indices]
        else:
            arm_indices = np.argmax(samples[:, :, 0], axis=1)
            return x_cand[arm_indices]
