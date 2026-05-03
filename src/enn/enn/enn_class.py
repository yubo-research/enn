from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from enn._rust import EpistemicNearestNeighbors as _RustENN
from enn.turbo.config.enn_index_driver import ENNIndexDriver

from .draw_internals import DrawInternals
from .enn_class_support import (
    _PosteriorMixin,
    _RustDispatchMixin,
    _compute_conditional_y_scale,
    _draw_from_internals,
    _rust_index_driver_name,
    _to_rust_seeds,
    enn_neighbor_distances_and_indices,
)

if TYPE_CHECKING:
    from .enn_normal import ENNNormal
    from .enn_params import ENNParams, PosteriorFlags


class EpistemicNearestNeighbors(_RustDispatchMixin, _PosteriorMixin):
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
        self._index_driver = index_driver
        self._x_scale = (
            self._compute_scale(self._train_x, 1e-12)
            if scale_x
            else np.ones((1, self._num_dim), dtype=float)
        )
        self._train_x_scaled = (
            self._train_x / self._x_scale if scale_x else self._train_x
        )
        self._y_scale = self._compute_scale(self._train_y, 0.0)
        idx_driver = _rust_index_driver_name(index_driver)
        self._rust_model = _RustENN(
            self._train_x,
            self._train_y,
            train_yvar=self._train_yvar,
            scale_x=scale_x,
            index_driver=idx_driver,
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
        self._y_scale = self._compute_scale(self._train_y, 0.0)
        if self._scale_x:
            self._x_scale = self._compute_scale(self._train_x, 1e-12)
            self._train_x_scaled = self._train_x / self._x_scale
        self._rust_model.add(x, y, yvar)

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

    @property
    def rust_backend(self):
        """Return the Rust backend model if available, otherwise None."""
        return self._rust_model

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

        def _rust_posterior(rust_model):
            mu, se, idx = rust_model.posterior(
                x,
                k_num_neighbors=params.k_num_neighbors,
                epistemic_variance_scale=params.epistemic_variance_scale,
                aleatoric_variance_scale=params.aleatoric_variance_scale,
                exclude_nearest=flags.exclude_nearest,
                observation_noise=flags.observation_noise,
            )
            idx_arr = np.array(idx, dtype=int) if idx else None
            return ENNNormal(mu, se, idx=idx_arr)

        def _python_posterior():
            internals = self._compute_posterior_internals(x, params, flags)
            return ENNNormal(internals.mu, internals.se, idx=internals.idx)

        return self._if_rust_else(_rust_posterior, _python_posterior)

    def conditional_posterior(
        self,
        x_whatif: np.ndarray,
        y_whatif: np.ndarray,
        x: np.ndarray,
        *,
        params: ENNParams,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal
        from .enn_params import PosteriorFlags

        if flags is None:
            flags = PosteriorFlags()

        def _rust_conditional(rust_model):
            mu, se, _ = rust_model.conditional_posterior(
                x_whatif,
                y_whatif,
                x,
                k_num_neighbors=params.k_num_neighbors,
                epistemic_variance_scale=params.epistemic_variance_scale,
                aleatoric_variance_scale=params.aleatoric_variance_scale,
                exclude_nearest=flags.exclude_nearest,
                observation_noise=flags.observation_noise,
            )
            return ENNNormal(mu, se)

        def _python_conditional():
            from .enn_conditional import compute_conditional_posterior

            y_scale = _compute_conditional_y_scale(self, y_whatif)
            return compute_conditional_posterior(
                self, x_whatif, y_whatif, x, params=params, flags=flags, y_scale=y_scale
            )

        return self._if_rust_else(_rust_conditional, _python_conditional)

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

        def _rust_batch(rust_model):
            k_values = [p.k_num_neighbors for p in paramss]
            epistemic_scales = [p.epistemic_variance_scale for p in paramss]
            aleatoric_scales = [p.aleatoric_variance_scale for p in paramss]
            mu_all, se_all = rust_model.batch_posterior(
                x,
                k_values=k_values,
                epistemic_scales=epistemic_scales,
                aleatoric_scales=aleatoric_scales,
                exclude_nearest=flags.exclude_nearest,
                observation_noise=flags.observation_noise,
            )
            return ENNNormal(mu_all, se_all)

        def _python_batch():
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

        return self._if_rust_else(_rust_batch, _python_batch)

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

        def _rust_neighbors(rust_model):
            idx_2d = rust_model.neighbors(x, k, exclude_nearest=exclude_nearest)
            idx = idx_2d[0, :] if idx_2d.size > 0 else np.array([], dtype=np.int64)
            return idx.astype(np.int64, copy=False)

        def _python_neighbors():
            search_k = int(min(k + 1 if exclude_nearest else k, len(self)))
            if search_k == 0:
                return np.zeros((0,), dtype=np.int64)
            _, idx_full = enn_neighbor_distances_and_indices(
                self._rust_model,
                x,
                search_k=search_k,
                exclude_nearest=exclude_nearest,
            )
            idx = idx_full[0, : min(k, idx_full.shape[1])]
            return idx.astype(np.int64, copy=False)

        return self._if_rust_else(_rust_neighbors, _python_neighbors)

    def posterior_function_draw(
        self,
        x: np.ndarray,
        params: ENNParams,
        *,
        function_seeds: np.ndarray | list[int],
        flags: PosteriorFlags | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from .enn_params import PosteriorFlags

        if flags is None:
            flags = PosteriorFlags()

        def _rust_draw(rust_model):
            seeds = _to_rust_seeds(function_seeds)
            draws, idx = rust_model.posterior_function_draw(
                x,
                k_num_neighbors=params.k_num_neighbors,
                epistemic_variance_scale=params.epistemic_variance_scale,
                aleatoric_variance_scale=params.aleatoric_variance_scale,
                function_seeds=seeds,
                exclude_nearest=flags.exclude_nearest,
                observation_noise=flags.observation_noise,
            )
            idx_arr = np.array(idx, dtype=int) if idx else np.zeros((x.shape[0], 0))
            return draws, idx_arr

        def _python_draw():
            internals = self._compute_posterior_internals(x, params, flags)
            return (
                _draw_from_internals(self, internals, function_seeds=function_seeds),
                internals.idx,
            )

        return self._if_rust_else(_rust_draw, _python_draw)

    def conditional_posterior_function_draw(
        self,
        x_whatif: np.ndarray,
        y_whatif: np.ndarray,
        x: np.ndarray,
        *,
        params: ENNParams,
        function_seeds: np.ndarray | list[int],
        flags: PosteriorFlags | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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

        def _rust_conditional_draw(rust_model):
            seeds = _to_rust_seeds(function_seeds)
            draws, idx = rust_model.conditional_posterior_function_draw(
                x_whatif,
                y_whatif,
                x,
                k_num_neighbors=params.k_num_neighbors,
                epistemic_variance_scale=params.epistemic_variance_scale,
                aleatoric_variance_scale=params.aleatoric_variance_scale,
                function_seeds=seeds,
                exclude_nearest=flags.exclude_nearest,
                observation_noise=flags.observation_noise,
            )
            idx_arr = np.array(idx, dtype=int) if idx else np.zeros((x.shape[0], 0))
            return draws, idx_arr

        def _python_conditional_draw():
            from .enn_conditional import compute_conditional_posterior_draw_internals

            y_scale = _compute_conditional_y_scale(self, y_whatif)
            internals = compute_conditional_posterior_draw_internals(
                self, x_whatif, y_whatif, x, params=params, flags=flags, y_scale=y_scale
            )
            draws = _draw_from_internals(
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
            return draws, internals.idx

        return self._if_rust_else(_rust_conditional_draw, _python_conditional_draw)
