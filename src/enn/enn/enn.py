from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from .enn_normal import ENNNormal
    from .enn_params import ENNParams


class EpistemicNearestNeighbors:
    _EPS_VAR = 1e-9

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_yvar: np.ndarray | None = None,
        *,
        scale_x: bool = False,
    ) -> None:
        import numpy as np

        train_x = np.asarray(train_x, dtype=float)
        train_y = np.asarray(train_y, dtype=float)
        if train_x.ndim != 2:
            raise ValueError(train_x.shape)
        if train_y.ndim != 2:
            raise ValueError(train_y.shape)
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError((train_x.shape, train_y.shape))
        if train_yvar is not None:
            train_yvar = np.asarray(train_yvar, dtype=float)
            if train_yvar.ndim != 2:
                raise ValueError(train_yvar.shape)
            if train_y.shape != train_yvar.shape:
                raise ValueError((train_y.shape, train_yvar.shape))

        self._train_x = train_x
        self._train_y = train_y
        self._train_yvar = train_yvar
        self._num_obs, self._num_dim = self._train_x.shape
        _, self._num_metrics = self._train_y.shape
        self._scale_x = bool(scale_x)
        if self._scale_x:
            if len(self._train_x) < 2:
                x_scale = np.ones((1, self._num_dim), dtype=float)
            else:
                x_scale = np.std(self._train_x, axis=0, keepdims=True).astype(float)
                x_scale = np.where(
                    np.isfinite(x_scale) & (x_scale > 1e-12),
                    x_scale,
                    1.0,
                )
            self._x_scale = x_scale
            self._train_x_scaled = self._train_x / self._x_scale
        else:
            self._x_scale = np.ones((1, self._num_dim), dtype=float)
            self._train_x_scaled = self._train_x
        if len(self._train_y) < 2:
            self._y_scale = np.ones(shape=(1, self._num_metrics), dtype=float)
        else:
            y_scale = np.std(self._train_y, axis=0, keepdims=True).astype(float)
            self._y_scale = np.where(
                np.isfinite(y_scale) & (y_scale > 0.0), y_scale, 1.0
            )

        from .enn_index import ENNIndex

        self._enn_index = ENNIndex(
            self._train_x_scaled, self._num_dim, self._x_scale, self._scale_x
        )

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

    def _search_index(
        self,
        x: np.ndarray,
        *,
        search_k: int,
        exclude_nearest: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._enn_index.search(
            x, search_k=search_k, exclude_nearest=exclude_nearest
        )

    def posterior(
        self,
        x: np.ndarray,
        *,
        params: ENNParams,
        exclude_nearest: bool = False,
        observation_noise: bool = False,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal

        post_batch = self.batch_posterior(
            x,
            [params],
            exclude_nearest=exclude_nearest,
            observation_noise=observation_noise,
        )
        mu = post_batch.mu[0]
        se = post_batch.se[0]
        return ENNNormal(mu, se)

    def _empty_posterior_internals(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import numpy as np

        mu = np.zeros((batch_size, self._num_metrics), dtype=float)
        se = np.ones((batch_size, self._num_metrics), dtype=float)
        idx = np.zeros((batch_size, 0), dtype=int)
        w_normalized = np.zeros((batch_size, 0, self._num_metrics), dtype=float)
        l2 = np.ones((batch_size, self._num_metrics), dtype=float)
        return idx, w_normalized, l2, mu, se

    def _get_neighbor_data(
        self, x: np.ndarray, params: ENNParams, exclude_nearest: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
        if exclude_nearest:
            if len(self) <= 1:
                raise ValueError(len(self))
            search_k = int(min(params.k + 1, len(self)))
        else:
            search_k = int(min(params.k, len(self)))
        dist2s_full, idx_full = self._search_index(
            x, search_k=search_k, exclude_nearest=exclude_nearest
        )
        available_k = search_k - 1 if exclude_nearest else search_k
        k = min(params.k, available_k)
        if k > dist2s_full.shape[1]:
            raise RuntimeError(
                f"k={k} exceeds available columns={dist2s_full.shape[1]}"
            )
        if k == 0:
            return None
        return dist2s_full[:, :k], idx_full[:, :k], self._train_y[idx_full[:, :k]], k

    def _compute_weighted_posterior(
        self,
        dist2s: np.ndarray,
        idx: np.ndarray,
        y_neighbors: np.ndarray,
        params: ENNParams,
        observation_noise: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import numpy as np

        dist2s_expanded = dist2s[..., np.newaxis]
        var_epi = params.epi_var_scale * dist2s_expanded
        var_ale = params.ale_homoscedastic_scale
        if self._train_yvar is not None:
            var_ale = var_ale + self._train_yvar[idx] / self._y_scale**2
        w = 1.0 / (self._EPS_VAR + var_epi + var_ale)
        norm = np.sum(w, axis=1, keepdims=True)
        w_normalized = w / norm
        l2 = np.sqrt(np.sum(w_normalized**2, axis=1))
        mu = np.sum(w_normalized * y_neighbors, axis=1)
        epistemic_var = 1.0 / norm.squeeze(axis=1)
        aleatoric_var = (
            np.sum(w_normalized * var_ale, axis=1) if observation_noise else 0.0
        )
        se = (
            np.sqrt(np.maximum(epistemic_var + aleatoric_var, self._EPS_VAR))
            * self._y_scale
        )
        return idx, w_normalized, l2, mu, se

    def _compute_posterior_internals(
        self,
        x: np.ndarray,
        params: ENNParams,
        *,
        exclude_nearest: bool = False,
        observation_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import numpy as np

        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        batch_size = x.shape[0]
        if len(self) == 0:
            return self._empty_posterior_internals(batch_size)
        neighbor_data = self._get_neighbor_data(x, params, exclude_nearest)
        if neighbor_data is None:
            return self._empty_posterior_internals(batch_size)
        dist2s, idx, y_neighbors, _ = neighbor_data
        return self._compute_weighted_posterior(
            dist2s, idx, y_neighbors, params, observation_noise
        )

    def batch_posterior(
        self,
        x: np.ndarray,
        paramss: list[ENNParams],
        *,
        exclude_nearest: bool = False,
        observation_noise: bool = False,
    ) -> ENNNormal:
        import numpy as np

        from .enn_normal import ENNNormal

        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(x.shape)
        if x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        if len(paramss) == 0:
            raise ValueError("paramss must be non-empty")
        batch_size = x.shape[0]
        num_params = len(paramss)
        mu_all = np.zeros((num_params, batch_size, self._num_metrics), dtype=float)
        se_all = np.zeros((num_params, batch_size, self._num_metrics), dtype=float)
        for i, params in enumerate(paramss):
            _, _, _, mu, se = self._compute_posterior_internals(
                x,
                params,
                exclude_nearest=exclude_nearest,
                observation_noise=observation_noise,
            )
            mu_all[i] = mu
            se_all[i] = se
        return ENNNormal(mu_all, se_all)

    def neighbors(
        self, x: np.ndarray, k: int, *, exclude_nearest: bool = False
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        import numpy as np

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
            return []
        if exclude_nearest and len(self) <= 1:
            raise ValueError(
                f"exclude_nearest=True requires at least 2 observations, got {len(self)}"
            )
        search_k = int(min(k + 1 if exclude_nearest else k, len(self)))
        if search_k == 0:
            return []
        _, idx_full = self._search_index(
            x, search_k=search_k, exclude_nearest=exclude_nearest
        )
        idx = idx_full[0, : min(k, len(idx_full[0]))]
        return [(self._train_x[i].copy(), self._train_y[i].copy()) for i in idx]

    def batch_posterior_function_sample(
        self,
        x: np.ndarray,
        params: ENNParams,
        *,
        function_seeds: np.ndarray | list[int],
        exclude_nearest: bool = False,
        observation_noise: bool = False,
    ) -> np.ndarray:
        import numpy as np
        from .enn_hash import normal_hash_batch_multi_seed

        function_seeds = np.asarray(function_seeds, dtype=np.int64)
        idx, w_normalized, l2, mu, se = self._compute_posterior_internals(
            x,
            params,
            exclude_nearest=exclude_nearest,
            observation_noise=observation_noise,
        )
        n, k, m = idx.shape[0], idx.shape[1], self._num_metrics
        if k == 0:
            return np.broadcast_to(mu, (len(function_seeds), n, m)).copy()
        u = normal_hash_batch_multi_seed(function_seeds, idx, m)
        weighted_u = np.sum(w_normalized[np.newaxis, :, :, :] * u, axis=2)
        l2_safe = np.maximum(l2, 1e-12)
        deviation = se[np.newaxis, :, :] * weighted_u / l2_safe[np.newaxis, :, :]
        return mu[np.newaxis, :, :] + deviation

    def posterior_function_sample(
        self,
        x: np.ndarray,
        params: ENNParams,
        *,
        function_seed: int,
        exclude_nearest: bool = False,
        observation_noise: bool = False,
    ) -> np.ndarray:
        return self.batch_posterior_function_sample(
            x,
            params,
            function_seeds=[function_seed],
            exclude_nearest=exclude_nearest,
            observation_noise=observation_noise,
        )[0]
