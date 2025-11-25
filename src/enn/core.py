from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from .enn_normal import ENNNormal
    from .enn_params import ENNParams


class EpistemicNearestNeighbors:
    def __init__(
        self,
        train_x: np.ndarray | Any,
        train_y: np.ndarray | Any,
        train_yvar: np.ndarray | Any,
        sobol_indices: bool = False,
    ) -> None:
        import numpy as np

        if train_x.ndim != 2:
            raise ValueError(train_x.shape)
        if train_y.ndim != 2 or train_yvar.ndim != 2:
            raise ValueError((train_y.shape, train_yvar.shape))
        if train_x.shape[0] != train_y.shape[0] or train_y.shape != train_yvar.shape:
            raise ValueError((train_x.shape, train_y.shape, train_yvar.shape))
        self._train_x = np.asarray(train_x, dtype=float)
        self._train_y = np.asarray(train_y, dtype=float)
        self._train_yvar = np.asarray(train_yvar, dtype=float)
        self._num_obs, self._num_dim = self._train_x.shape
        _, self._num_metrics = self._train_y.shape
        self._eps_var = 1e-9
        self._x_scale = self._calc_x_scale(sobol_indices)
        if len(self._train_y) < 2:
            self._y_scale = np.ones(shape=(1, self._num_metrics), dtype=float)
        else:
            self._y_scale = np.std(self._train_y, axis=0, keepdims=True).astype(float)

        self._index: Any | None = None
        self._build_index()

    @property
    def train_x(self) -> np.ndarray:
        return self._train_x

    @property
    def train_y(self) -> np.ndarray:
        return self._train_y

    @property
    def train_yvar(self) -> np.ndarray:
        return self._train_yvar

    @property
    def num_outputs(self) -> int:
        return self._num_metrics

    def __len__(self) -> int:
        return self._num_obs

    def _calc_x_scale(self, sobol_indices: bool) -> np.ndarray:
        import numpy as np

        s = np.std(self._train_x, axis=0, ddof=0).astype(float)
        s = np.maximum(s, 1e-6)
        if sobol_indices:
            from .enn_util import calculate_sobol_indices

            if self._num_metrics != 1:
                raise ValueError(
                    "sobol_indices=True requires train_y to have exactly 1 metric"
                )
            si = calculate_sobol_indices(self._train_x, self._train_y[:, 0])
            w = si / s
            w_sum = np.maximum(w.sum(), 1e-12)
            w = w / w_sum
            return s / w
        else:
            return s

    def _build_index(self) -> None:
        import faiss
        import numpy as np

        if self._num_obs == 0:
            return
        x_scaled = self._train_x / self._x_scale
        x_scaled = x_scaled.astype(np.float32, copy=False)
        index = faiss.IndexFlatL2(self._num_dim)
        index.add(x_scaled)
        self._index = index

    def posterior(
        self,
        x: np.ndarray | Any,
        *,
        params: ENNParams,
        exclude_nearest: bool = False,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal

        post_batch = self.batch_posterior(x, [params], exclude_nearest=exclude_nearest)
        mu = post_batch.mu[0]
        se = post_batch.se[0]
        return ENNNormal(mu, se)

    def batch_posterior(
        self,
        x: np.ndarray | Any,
        paramss: list[ENNParams],
        *,
        exclude_nearest: bool = False,
    ) -> ENNNormal:
        import numpy as np

        from .enn_normal import ENNNormal

        if x.ndim != 2:
            raise ValueError(x.shape)
        if x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        if len(paramss) == 0:
            raise ValueError("paramss must be non-empty")
        batch_size = x.shape[0]
        num_params = len(paramss)
        if len(self) == 0:
            mu = np.zeros((num_params, batch_size, self._num_metrics), dtype=float)
            se = np.ones((num_params, batch_size, self._num_metrics), dtype=float)
            return ENNNormal(mu, se)
        max_k = max(params.k for params in paramss)
        if exclude_nearest:
            if len(self) <= 1:
                raise ValueError(len(self))
            search_k = int(min(max_k + 1, len(self)))
        else:
            search_k = int(min(max_k, len(self)))
        x_scaled = x / self._x_scale
        x_scaled = x_scaled.astype(np.float32, copy=False)
        if self._index is None:
            raise RuntimeError("index is not initialized")
        dist2s_full, idx_full = self._index.search(x_scaled, search_k)
        dist2s_full = dist2s_full.astype(float)
        idx_full = idx_full.astype(int)
        if exclude_nearest:
            dist2s_full = dist2s_full[:, 1:]
            idx_full = idx_full[:, 1:]
        mu_all = np.zeros((num_params, batch_size, self._num_metrics), dtype=float)
        se_all = np.zeros((num_params, batch_size, self._num_metrics), dtype=float)
        for i, params in enumerate(paramss):
            k = min(params.k, search_k)
            if k == 0:
                mu_all[i] = np.zeros((batch_size, self._num_metrics), dtype=float)
                se_all[i] = np.ones((batch_size, self._num_metrics), dtype=float)
                continue
            dist2s = dist2s_full[:, :k]
            idx = idx_full[:, :k]
            y_neighbors = self._train_y[idx]
            yvar_neighbors = self._train_yvar[idx]
            dist2s_expanded = dist2s[..., np.newaxis]
            var_component = (
                params.var_scale * dist2s_expanded + yvar_neighbors / self._y_scale**2
            )
            w = 1.0 / (self._eps_var + var_component)
            norm = np.sum(w, axis=1)
            mu_all[i] = np.sum(w * y_neighbors, axis=1) / norm
            epistemic_var = 1.0 / norm
            noise_var = np.sum(w * yvar_neighbors, axis=1) / norm
            vvar = epistemic_var + noise_var
            vvar = np.maximum(vvar, self._eps_var)
            se_all[i] = np.sqrt(vvar) * self._y_scale
        return ENNNormal(mu_all, se_all)

    def neighbors(
        self,
        x: np.ndarray | Any,
        k: int,
        *,
        exclude_nearest: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        import numpy as np

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.ndim != 2:
            raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")
        if x.shape[0] != 1:
            raise ValueError(f"x must be a single point, got shape {x.shape}")
        if x.shape[1] != self._num_dim:
            raise ValueError(
                f"x must have {self._num_dim} dimensions, got {x.shape[1]}"
            )
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if len(self) == 0:
            return []
        if exclude_nearest:
            if len(self) <= 1:
                raise ValueError(
                    f"exclude_nearest=True requires at least 2 observations, got {len(self)}"
                )
            search_k = int(min(k + 1, len(self)))
        else:
            search_k = int(min(k, len(self)))
        if search_k == 0:
            return []
        x_scaled = x / self._x_scale
        x_scaled = x_scaled.astype(np.float32, copy=False)
        if self._index is None:
            raise RuntimeError("index is not initialized")
        dist2s_full, idx_full = self._index.search(x_scaled, search_k)
        dist2s_full = dist2s_full.astype(float)
        idx_full = idx_full.astype(int)
        if exclude_nearest:
            dist2s_full = dist2s_full[:, 1:]
            idx_full = idx_full[:, 1:]
        actual_k = min(k, len(idx_full[0]))
        idx = idx_full[0, :actual_k]
        result = []
        for i in idx:
            x_neighbor = self._train_x[i].copy()
            y_neighbor = self._train_y[i].copy()
            result.append((x_neighbor, y_neighbor))
        return result
