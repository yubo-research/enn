from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .turbo_utils_core import torch_seed_context

if TYPE_CHECKING:
    from numpy.random import Generator

__all__ = ["gp_thompson_sample"]


def gp_thompson_sample(
    model: Any,
    x_cand: np.ndarray | Any,
    num_arms: int,
    rng: Generator | Any,
    *,
    gp_y_mean: float,
    gp_y_std: float,
) -> np.ndarray:
    import gpytorch
    import torch

    x_torch = torch.as_tensor(x_cand, dtype=torch.float64)
    seed = int(rng.integers(2**31 - 1))
    with (
        torch.no_grad(),
        gpytorch.settings.fast_pred_var(),
        torch_seed_context(seed, device=x_torch.device),
    ):
        posterior = model.posterior(x_torch)
        samples = posterior.sample(sample_shape=torch.Size([1]))
    if samples.ndim != 2:
        raise ValueError(samples.shape)
    ts = samples[0].reshape(-1)
    scores = ts.detach().cpu().numpy().reshape(-1)
    scores = gp_y_mean + gp_y_std * scores
    shuffled_indices = rng.permutation(len(scores))
    shuffled_scores = scores[shuffled_indices]
    top_k_in_shuffled = np.argpartition(-shuffled_scores, num_arms - 1)[:num_arms]
    idx = shuffled_indices[top_k_in_shuffled]
    return idx
