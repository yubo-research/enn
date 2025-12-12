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
        train_yvar: np.ndarray | Any | None = None,
    ) -> None:
        import numpy as np

        if train_x.ndim != 2:
            raise ValueError(train_x.shape)
        if train_y.ndim != 2:
            raise ValueError(train_y.shape)
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError((train_x.shape, train_y.shape))
        if train_yvar is not None:
            if train_yvar.ndim != 2:
                raise ValueError(train_yvar.shape)
            if train_y.shape != train_yvar.shape:
                raise ValueError((train_y.shape, train_yvar.shape))
        self._train_x = np.asarray(train_x, dtype=float)
        self._train_y = np.asarray(train_y, dtype=float)
        self._train_yvar = (
            np.asarray(train_yvar, dtype=float) if train_yvar is not None else None
        )
        self._num_obs, self._num_dim = self._train_x.shape
        _, self._num_metrics = self._train_y.shape
        self._eps_var = 1e-9
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
    def train_yvar(self) -> np.ndarray | None:
        return self._train_yvar

    @property
    def num_outputs(self) -> int:
        return self._num_metrics

    def __len__(self) -> int:
        return self._num_obs

    def _build_index(self) -> None:
        import faiss
        import numpy as np

        if self._num_obs == 0:
            return
        x_f32 = self._train_x.astype(np.float32, copy=False)
        index = faiss.IndexFlatL2(self._num_dim)
        index.add(x_f32)
        self._index = index

    def posterior(
        self,
        x: np.ndarray | Any,
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

    def batch_posterior(
        self,
        x: np.ndarray | Any,
        paramss: list[ENNParams],
        *,
        exclude_nearest: bool = False,
        observation_noise: bool = False,
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
        x_f32 = x.astype(np.float32, copy=False)
        if self._index is None:
            raise RuntimeError("index is not initialized")
        dist2s_full, idx_full = self._index.search(x_f32, search_k)
        dist2s_full = dist2s_full.astype(float)
        idx_full = idx_full.astype(int)
        if exclude_nearest:
            dist2s_full = dist2s_full[:, 1:]
            idx_full = idx_full[:, 1:]
        mu_all = np.zeros((num_params, batch_size, self._num_metrics), dtype=float)
        se_all = np.zeros((num_params, batch_size, self._num_metrics), dtype=float)
        available_k = search_k - 1 if exclude_nearest else search_k
        for i, params in enumerate(paramss):
            k = min(params.k, available_k)
            if k > dist2s_full.shape[1]:
                raise RuntimeError(
                    f"k={k} exceeds available columns={dist2s_full.shape[1]}"
                )
            if k == 0:
                mu_all[i] = np.zeros((batch_size, self._num_metrics), dtype=float)
                se_all[i] = np.ones((batch_size, self._num_metrics), dtype=float)
                continue
            dist2s = dist2s_full[:, :k]
            idx = idx_full[:, :k]
            y_neighbors = self._train_y[idx]

            dist2s_expanded = dist2s[..., np.newaxis]
            var_component = (
                params.ale_homoscedastic_scale + params.epi_var_scale * dist2s_expanded
            )
            if self._train_yvar is not None:
                yvar_neighbors = self._train_yvar[idx] / self._y_scale**2
                var_component = var_component + yvar_neighbors
            else:
                yvar_neighbors = None

            w = 1.0 / (self._eps_var + var_component)
            norm = np.sum(w, axis=1)
            mu_all[i] = np.sum(w * y_neighbors, axis=1) / norm
            epistemic_var = 1.0 / norm
            vvar = epistemic_var
            if observation_noise:
                vvar = vvar + params.ale_homoscedastic_scale
                if yvar_neighbors is not None:
                    ale_heteroscedastic = np.sum(w * yvar_neighbors, axis=1) / norm
                    vvar = vvar + ale_heteroscedastic
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
        x_f32 = x.astype(np.float32, copy=False)
        if self._index is None:
            raise RuntimeError("index is not initialized")
        dist2s_full, idx_full = self._index.search(x_f32, search_k)
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
