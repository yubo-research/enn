from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from enn.turbo.config.enn_index_driver import ENNIndexDriver

from .draw_internals import DrawInternals
from .neighbor_data import NeighborData
from .weighted_stats import WeightedStats

if TYPE_CHECKING:
    from .enn_normal import ENNNormal
    from .enn_params import ENNParams, PosteriorFlags


def _compute_conditional_y_scale(
    model: EpistemicNearestNeighbors, y_whatif: np.ndarray
):
    y_whatif = np.asarray(y_whatif, dtype=float)
    return model._compute_scale(
        np.concatenate([model.train_y, y_whatif], axis=0),
        0.0,
    )


def _draw_from_internals(
    model: EpistemicNearestNeighbors,
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


class EpistemicNearestNeighbors:
    _EPS_VAR = 1e-9

    @staticmethod
    def _validate_inputs(train_x, train_y, train_yvar):
        train_x, train_y = (
            np.asarray(train_x, dtype=float),
            np.asarray(train_y, dtype=float),
        )
        if (
            train_x.ndim != 2
            or train_y.ndim != 2
            or train_x.shape[0] != train_y.shape[0]
        ):
            raise ValueError((train_x.shape, train_y.shape))
        if train_yvar is not None:
            train_yvar = np.asarray(train_yvar, dtype=float)
            if train_yvar.ndim != 2 or train_y.shape != train_yvar.shape:
                raise ValueError((train_y.shape, train_yvar.shape))
        return train_x, train_y, train_yvar

    @staticmethod
    def _compute_scale(data, min_val=0.0):
        if len(data) < 2:
            return np.ones((1, data.shape[1]), dtype=float)
        scale = np.std(data, axis=0, keepdims=True).astype(float)
        return np.where(np.isfinite(scale) & (scale > min_val), scale, 1.0)

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_yvar: np.ndarray | None = None,
        *,
        scale_x: bool = False,
        index_driver: ENNIndexDriver = ENNIndexDriver.FLAT,
    ) -> None:
        self._train_x, self._train_y, self._train_yvar = self._validate_inputs(
            train_x, train_y, train_yvar
        )
        self._num_obs, self._num_dim = self._train_x.shape
        _, self._num_metrics = self._train_y.shape
        self._scale_x = bool(scale_x)
        self._x_scale = (
            self._compute_scale(self._train_x, 1e-12)
            if scale_x
            else np.ones((1, self._num_dim), dtype=float)
        )
        self._train_x_scaled = (
            self._train_x / self._x_scale if scale_x else self._train_x
        )
        self._y_scale = self._compute_scale(self._train_y, 0.0)
        from .enn_index import ENNIndex

        self._enn_index = ENNIndex(
            self._train_x_scaled,
            self._num_dim,
            self._x_scale,
            self._scale_x,
            driver=index_driver,
        )

    def add(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yvar: np.ndarray | None = None,
    ) -> None:
        x, y, yvar = self._validate_inputs(x, y, yvar)
        self._train_x = np.concatenate([self._train_x, x], axis=0)
        self._train_y = np.concatenate([self._train_y, y], axis=0)
        if yvar is not None:
            if self._train_yvar is None:
                self._train_yvar = yvar
            else:
                self._train_yvar = np.concatenate([self._train_yvar, yvar], axis=0)
        elif self._train_yvar is not None:
            # If we have some yvar but not for the new points, we need to handle it.
            # For now, we'll just use zeros or raise if inconsistent.
            # Following the existing pattern, we assume consistency.
            raise ValueError("yvar must be provided if model has existing yvar")

        self._num_obs = self._train_x.shape[0]
        self._enn_index.add(x)

    @property
    def train_x(self) -> np.ndarray:
        return self._train_x

    @property
    def train_y(self) -> np.ndarray:
        return self._train_y

    @property
    def train_yvar(self) -> np.ndarray | None:
        return self._train_yvar

    @property
    def num_outputs(self) -> int:
        return self._num_metrics

    def __len__(self) -> int:
        return self._num_obs

    def posterior(
        self,
        x: np.ndarray,
        *,
        params: ENNParams,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal
        from .enn_params import PosteriorFlags

        if flags is None:
            flags = PosteriorFlags()
        post_batch = self.batch_posterior(x, [params], flags=flags)
        return ENNNormal(post_batch.mu[0], post_batch.se[0])

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
        self, x: np.ndarray, params: ENNParams, exclude_nearest: bool
    ) -> NeighborData | None:
        if exclude_nearest:
            if len(self) <= 1:
                raise ValueError(len(self))
            search_k = int(min(params.k_num_neighbors + 1, len(self)))
        else:
            search_k = int(min(params.k_num_neighbors, len(self)))
        dist2s_full, idx_full = self._enn_index.search(
            x, search_k=search_k, exclude_nearest=exclude_nearest
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
        params: ENNParams,
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
        params: ENNParams,
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

    def conditional_posterior(
        self,
        x_whatif: np.ndarray,
        y_whatif: np.ndarray,
        x: np.ndarray,
        *,
        params: ENNParams,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        from .enn_conditional import compute_conditional_posterior
        from .enn_params import PosteriorFlags

        if flags is None:
            flags = PosteriorFlags()
        y_scale = _compute_conditional_y_scale(self, y_whatif)
        return compute_conditional_posterior(
            self, x_whatif, y_whatif, x, params=params, flags=flags, y_scale=y_scale
        )

    def _compute_posterior_internals(
        self,
        x: np.ndarray,
        params: ENNParams,
        flags: PosteriorFlags,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def batch_posterior(
        self,
        x: np.ndarray,
        paramss: list[ENNParams],
        *,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal
        from .enn_params import PosteriorFlags

        if flags is None:
            flags = PosteriorFlags()
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        if not paramss:
            raise ValueError("paramss must be non-empty")
        batch_size, num_params = x.shape[0], len(paramss)
        mu_all = np.zeros((num_params, batch_size, self._num_metrics), dtype=float)
        se_all = np.zeros((num_params, batch_size, self._num_metrics), dtype=float)
        k_values = {p.k_num_neighbors for p in paramss}
        if len(k_values) == 1 and len(self) > 0:
            neighbor_data = self._get_neighbor_data(
                x, paramss[0], flags.exclude_nearest
            )
            if neighbor_data is None:
                return ENNNormal(mu_all, se_all)
            for i, params in enumerate(paramss):
                internals = self._compute_weighted_posterior(
                    neighbor_data.dist2s,
                    neighbor_data.idx,
                    neighbor_data.y_neighbors,
                    params,
                    flags.observation_noise,
                )
                mu_all[i], se_all[i] = internals.mu, internals.se
        else:
            for i, params in enumerate(paramss):
                internals = self._compute_posterior_internals(x, params, flags)
                mu_all[i], se_all[i] = internals.mu, internals.se
        return ENNNormal(mu_all, se_all)

    def neighbors(
        self, x: np.ndarray, k: int, *, exclude_nearest: bool = False
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] != self._num_dim:
            raise ValueError(
                f"x must be single point with {self._num_dim} dims, got {x.shape}"
            )
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if len(self) == 0:
            return np.zeros((0,), dtype=np.int64)
        if exclude_nearest and len(self) <= 1:
            raise ValueError(
                f"exclude_nearest=True requires at least 2 observations, got {len(self)}"
            )
        search_k = int(min(k + 1 if exclude_nearest else k, len(self)))
        if search_k == 0:
            return np.zeros((0,), dtype=np.int64)
        _, idx_full = self._enn_index.search(
            x, search_k=search_k, exclude_nearest=exclude_nearest
        )
        idx = idx_full[0, : min(k, idx_full.shape[1])]
        return idx.astype(np.int64, copy=False)

    def posterior_function_draw(
        self,
        x: np.ndarray,
        params: ENNParams,
        *,
        function_seeds: np.ndarray | list[int],
        flags: PosteriorFlags | None = None,
    ) -> np.ndarray:
        from .enn_params import PosteriorFlags

        if flags is None:
            flags = PosteriorFlags()
        internals = self._compute_posterior_internals(x, params, flags)
        return _draw_from_internals(self, internals, function_seeds=function_seeds)

    def conditional_posterior_function_draw(
        self,
        x_whatif: np.ndarray,
        y_whatif: np.ndarray,
        x: np.ndarray,
        *,
        params: ENNParams,
        function_seeds: np.ndarray | list[int],
        flags: PosteriorFlags | None = None,
    ) -> np.ndarray:
        from .enn_conditional import compute_conditional_posterior_draw_internals
        from .enn_params import PosteriorFlags

        if flags is None:
            flags = PosteriorFlags()
        x_whatif = np.asarray(x_whatif, dtype=float)
        if x_whatif.ndim != 2 or x_whatif.shape[1] != self._num_dim:
            raise ValueError(x_whatif.shape)
        if x_whatif.shape[0] == 0:
            return self.posterior_function_draw(
                x,
                params,
                function_seeds=function_seeds,
                flags=flags,
            )
        y_scale = _compute_conditional_y_scale(self, y_whatif)
        internals = compute_conditional_posterior_draw_internals(
            self, x_whatif, y_whatif, x, params=params, flags=flags, y_scale=y_scale
        )
        return _draw_from_internals(
            self,
            DrawInternals(
                idx=internals.idx,
                w_normalized=internals.w_normalized,
                l2=internals.l2,
                mu=internals.mu,
                se=internals.se,
            ),
            function_seeds=function_seeds,
        )
