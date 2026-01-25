from __future__ import annotations
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    from .enn_params import ENNParams, PosteriorFlags


class ENNLike(Protocol):
    _num_dim: int
    _num_metrics: int
    _x_scale: np.ndarray
    _scale_x: bool
    _enn_index: Any
    _train_y: np.ndarray
    _train_yvar: np.ndarray | None

    def __len__(self) -> int: ...
    def posterior(self, x: np.ndarray, *, params: ENNParams, flags: PosteriorFlags):
        raise NotImplementedError

    def _empty_posterior_internals(self, batch_size: int):
        raise NotImplementedError

    def _compute_weighted_stats(
        self,
        dist2s: np.ndarray,
        y_neighbors: np.ndarray,
        *,
        yvar_neighbors: np.ndarray | None,
        params: ENNParams,
        observation_noise: bool,
        y_scale: np.ndarray | None = None,
    ):
        raise NotImplementedError
