from __future__ import annotations

from typing import Protocol

import numpy as np

from enn.turbo.config.enn_index_driver import ENNIndexDriver

from .draw_internals import DrawInternals
from .neighbor_data import NeighborData
from .weighted_stats import WeightedStats


class _SupportsConditionalScale(Protocol):
    train_y: np.ndarray

    def _compute_scale(self, data: np.ndarray, min_val: float) -> np.ndarray: ...


class _SupportsDrawFromInternals(Protocol):
    num_outputs: int


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


def _compute_conditional_y_scale(
    model: _SupportsConditionalScale, y_whatif: np.ndarray
) -> np.ndarray:
    y_whatif = np.asarray(y_whatif, dtype=float)
    return model._compute_scale(
        np.concatenate([model.train_y, y_whatif], axis=0),
        0.0,
    )


def _draw_from_internals(
    model: _SupportsDrawFromInternals,
    internals: DrawInternals,
    *,
    function_seeds: np.ndarray | list[int],
) -> np.ndarray:
    from .enn_hash import normal_hash_batch_multi_seed_fast

    function_seeds = np.asarray(function_seeds, dtype=np.int64)
    n, k, m = internals.idx.shape[0], internals.idx.shape[1], model.num_outputs
    if k == 0:
        return np.broadcast_to(internals.mu, (len(function_seeds), n, m)).copy()
    u = normal_hash_batch_multi_seed_fast(function_seeds, internals.idx, m)
    weighted_u = np.sum(internals.w_normalized[np.newaxis, :, :, :] * u, axis=2)
    l2_safe = np.maximum(internals.l2, 1e-12)
    return (
        internals.mu[np.newaxis, :, :]
        + internals.se[np.newaxis, :, :] * weighted_u / l2_safe[np.newaxis, :, :]
    )


class _RustDispatchMixin:
    """Mixin providing Rust/Python dispatch helpers.

    This mixin must come FIRST in the inheritance list to ensure
    proper MRO when combined with other mixins.
    """

    def _if_rust_else(
        self,
        rust_fn: callable,
        python_fn: callable,
        *,
        check_rust: bool = True,
    ):
        """Execute rust_fn if Rust backend is available, otherwise python_fn."""
        if check_rust and self._rust_model is not None:
            return rust_fn(self._rust_model)
        return python_fn()


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

    def _get_neighbor_data(
        self, x: np.ndarray, params, exclude_nearest: bool
    ) -> NeighborData | None:
        if exclude_nearest:
            if len(self) <= 1:
                raise ValueError(len(self))
            search_k = int(min(params.k_num_neighbors + 1, len(self)))
        else:
            search_k = int(min(params.k_num_neighbors, len(self)))
        dist2s_full, idx_full = enn_neighbor_distances_and_indices(
            self._rust_model,
            x,
            search_k=search_k,
            exclude_nearest=exclude_nearest,
        )
        available_k = search_k - 1 if exclude_nearest else search_k
        k = min(params.k_num_neighbors, available_k)
        if k > dist2s_full.shape[1]:
            raise RuntimeError(
                f"k={k} exceeds available columns={dist2s_full.shape[1]}"
            )
        if k == 0:
            return None
        return NeighborData(
            dist2s=dist2s_full[:, :k],
            idx=idx_full[:, :k],
            y_neighbors=self._train_y[idx_full[:, :k]],
            k=k,
        )

    def _compute_weighted_posterior(
        self,
        dist2s: np.ndarray,
        idx: np.ndarray,
        y_neighbors: np.ndarray,
        params,
        observation_noise: bool,
    ) -> DrawInternals:
        yvar_neighbors = None
        if self._train_yvar is not None:
            yvar_neighbors = self._train_yvar[idx]
        stats = self._compute_weighted_stats(
            dist2s,
            y_neighbors,
            yvar_neighbors=yvar_neighbors,
            params=params,
            observation_noise=observation_noise,
        )
        return DrawInternals(
            idx=idx,
            w_normalized=stats.w_normalized,
            l2=stats.l2,
            mu=stats.mu,
            se=stats.se,
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

    def _compute_posterior_internals(self, x, params, flags) -> DrawInternals:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        batch_size = x.shape[0]
        if len(self) == 0:
            return self._empty_posterior_internals(batch_size)
        neighbor_data = self._get_neighbor_data(x, params, flags.exclude_nearest)
        if neighbor_data is None:
            return self._empty_posterior_internals(batch_size)
        return self._compute_weighted_posterior(
            neighbor_data.dist2s,
            neighbor_data.idx,
            neighbor_data.y_neighbors,
            params,
            flags.observation_noise,
        )
