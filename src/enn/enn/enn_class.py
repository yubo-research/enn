from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from enn._rust import EpistemicNearestNeighbors as _RustENN
from enn.turbo.config.enn_index_driver import ENNIndexDriver

from .enn_class_support import _rust_index_driver_name, _to_rust_seeds

if TYPE_CHECKING:
    from .enn_normal import ENNNormal
    from .enn_params import ENNParams, PosteriorFlags


def _posterior_flags_coerced(flags):
    from .enn_params import PosteriorFlags

    return flags if flags is not None else PosteriorFlags()


def _rust_function_draw_kwargs(params, flags, seeds: list[int]) -> dict[str, Any]:
    return {
        "k_num_neighbors": params.k_num_neighbors,
        "epistemic_variance_scale": params.epistemic_variance_scale,
        "aleatoric_variance_scale": params.aleatoric_variance_scale,
        "function_seeds": seeds,
        "exclude_nearest": flags.exclude_nearest,
        "observation_noise": flags.observation_noise,
    }


def _finalize_function_draw(
    x: np.ndarray, draws: np.ndarray, idx: list | None
) -> tuple[np.ndarray, np.ndarray]:
    idx_arr = np.array(idx, dtype=int) if idx else np.zeros((x.shape[0], 0), dtype=int)
    return draws, idx_arr


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

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_yvar: np.ndarray | None = None,
        *,
        scale_x: bool = False,
        index_driver: ENNIndexDriver = ENNIndexDriver.FLAT,
    ) -> None:
        train_x, train_y, train_yvar = self._validate_inputs(
            train_x, train_y, train_yvar
        )
        self._index_driver = index_driver
        idx_driver = _rust_index_driver_name(index_driver)
        self._rust_model = _RustENN(
            train_x,
            train_y,
            train_yvar=train_yvar,
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
        self._rust_model.add(x, y, yvar)

    @property
    def train_x(self) -> np.ndarray:
        return np.asarray(self._rust_model.train_x, dtype=float)

    @property
    def train_y(self) -> np.ndarray:
        return np.asarray(self._rust_model.train_y, dtype=float)

    @property
    def train_yvar(self) -> np.ndarray | None:
        tyv = self._rust_model.train_yvar
        if tyv is None:
            return None
        return np.asarray(tyv, dtype=float)

    @property
    def num_outputs(self) -> int:
        return int(self._rust_model.num_outputs)

    @property
    def rust_backend(self):
        """Rust surrogate implementation (same lifetime as this wrapper)."""
        return self._rust_model

    @property
    def _num_dim(self) -> int:
        return int(self._rust_model.num_dim)

    @property
    def _num_metrics(self) -> int:
        return int(self._rust_model.num_outputs)

    @property
    def _x_scale(self) -> np.ndarray:
        return np.asarray(self._rust_model.x_scale_row, dtype=float)

    @property
    def _y_scale(self) -> np.ndarray:
        return np.asarray(self._rust_model.y_scale_row, dtype=float)

    @property
    def _scale_x(self) -> bool:
        return bool(self._rust_model.scale_x)

    @property
    def _train_y(self) -> np.ndarray:
        return self.train_y

    @property
    def _train_yvar(self) -> np.ndarray | None:
        return self.train_yvar

    def __len__(self) -> int:
        return len(self._rust_model)

    def posterior(
        self,
        x: np.ndarray,
        *,
        params: ENNParams,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal

        flags = _posterior_flags_coerced(flags)

        mu, se, idx = self._rust_model.posterior(
            x,
            k_num_neighbors=params.k_num_neighbors,
            epistemic_variance_scale=params.epistemic_variance_scale,
            aleatoric_variance_scale=params.aleatoric_variance_scale,
            exclude_nearest=flags.exclude_nearest,
            observation_noise=flags.observation_noise,
        )
        idx_arr = np.array(idx, dtype=int) if idx else None
        return ENNNormal(mu, se, idx=idx_arr)

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

        flags = _posterior_flags_coerced(flags)

        mu, se, _ = self._rust_model.conditional_posterior(
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

    def batch_posterior(
        self,
        x: np.ndarray,
        paramss: list[ENNParams],
        *,
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal

        flags = _posterior_flags_coerced(flags)
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        if not paramss:
            raise ValueError("paramss must be non-empty")

        k_values = [p.k_num_neighbors for p in paramss]
        epistemic_scales = [p.epistemic_variance_scale for p in paramss]
        aleatoric_scales = [p.aleatoric_variance_scale for p in paramss]
        mu_all, se_all = self._rust_model.batch_posterior(
            x,
            k_values=k_values,
            epistemic_scales=epistemic_scales,
            aleatoric_scales=aleatoric_scales,
            exclude_nearest=flags.exclude_nearest,
            observation_noise=flags.observation_noise,
        )
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

        idx_2d = self._rust_model.neighbors(x, k, exclude_nearest=exclude_nearest)
        idx = idx_2d[0, :] if idx_2d.size > 0 else np.array([], dtype=np.int64)
        return idx.astype(np.int64, copy=False)

    def posterior_function_draw(
        self,
        x: np.ndarray,
        params: ENNParams,
        *,
        function_seeds: np.ndarray | list[int],
        flags: PosteriorFlags | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        flags = _posterior_flags_coerced(flags)
        seeds = _to_rust_seeds(function_seeds)
        kw = _rust_function_draw_kwargs(params, flags, seeds)
        draws, idx = self._rust_model.posterior_function_draw(x, **kw)
        return _finalize_function_draw(x, draws, idx)

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
        flags = _posterior_flags_coerced(flags)
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

        seeds = _to_rust_seeds(function_seeds)
        kw = _rust_function_draw_kwargs(params, flags, seeds)
        draws, idx = self._rust_model.conditional_posterior_function_draw(
            x_whatif,
            y_whatif,
            x,
            **kw,
        )
        return _finalize_function_draw(x, draws, idx)
