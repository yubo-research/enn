from __future__ import annotations

import numpy as np

from enn.turbo.config.enn_index_driver import ENNIndexDriver

from .draw_internals import DrawInternals
from .weighted_stats import WeightedStats


def _to_rust_seeds(function_seeds: np.ndarray | list[int]) -> list[int]:
    """Convert function_seeds to Rust list[int] format."""
    if hasattr(function_seeds, "__iter__"):
        return np.asarray(function_seeds, dtype=np.int64).tolist()
    return list(function_seeds)


def enn_neighbor_distances_and_indices(
    rust_model,
    x: np.ndarray,
    *,
    search_k: int,
    exclude_nearest: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dist2s, idx = rust_model.neighbor_distances_and_indices(
        np.asarray(x, dtype=float),
        int(search_k),
        bool(exclude_nearest),
    )
    return np.asarray(dist2s, dtype=float), np.asarray(idx, dtype=int)


def _rust_index_driver_name(index_driver: ENNIndexDriver) -> str:
    from enn.turbo.config.enn_index_driver import ENN_INDEX_DRIVER_TO_RUST

    if index_driver not in ENN_INDEX_DRIVER_TO_RUST:
        raise ValueError(f"Unsupported index driver: {index_driver}")
    return ENN_INDEX_DRIVER_TO_RUST[index_driver]


class _PosteriorMixin:
    """Mixin for posterior computation helpers."""

    def _empty_posterior_internals(self, batch_size: int) -> DrawInternals:
        m = self._num_metrics
        return DrawInternals(
            idx=np.zeros((batch_size, 0), dtype=int),
            w_normalized=np.zeros((batch_size, 0, m), dtype=float),
            l2=np.ones((batch_size, m), dtype=float),
            mu=np.zeros((batch_size, m), dtype=float),
            se=np.ones((batch_size, m), dtype=float),
        )

    def _compute_weighted_stats(
        self,
        dist2s: np.ndarray,
        y_neighbors: np.ndarray,
        *,
        yvar_neighbors: np.ndarray | None,
        params,
        observation_noise: bool,
        y_scale: np.ndarray | None = None,
    ) -> WeightedStats:
        if y_scale is None:
            y_scale = self._y_scale
        dist2s_expanded = dist2s[..., np.newaxis]
        var_epi = params.epistemic_variance_scale * dist2s_expanded
        var_ale = params.aleatoric_variance_scale
        if yvar_neighbors is not None:
            var_ale = var_ale + yvar_neighbors / y_scale**2
        w = 1.0 / (self._EPS_VAR + var_epi + var_ale)
        norm = np.sum(w, axis=1, keepdims=True)
        w_normalized = w / norm
        l2 = np.sqrt(np.sum(w_normalized**2, axis=1))
        mu = np.sum(w_normalized * y_neighbors, axis=1)
        epistemic_var = 1.0 / norm.squeeze(axis=1)
        if observation_noise:
            if np.isscalar(var_ale):
                aleatoric_var = np.full_like(epistemic_var, var_ale)
            else:
                aleatoric_var = np.sum(w_normalized * var_ale, axis=1)
        else:
            aleatoric_var = 0.0
        se = np.sqrt(np.maximum(epistemic_var + aleatoric_var, self._EPS_VAR)) * y_scale
        return WeightedStats(w_normalized=w_normalized, l2=l2, mu=mu, se=se)
