from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from enn._rust import ENNAddToken, InputHistory
from enn._rust import EpistemicNearestNeighbors as _RustENN
from enn.turbo.config.enn_index_driver import (
    ENNIndexDriver,
    index_driver_from_wire,
    index_driver_to_wire,
)
from enn.turbo.config.enn_surrogate_config import validate_enn_placement
from enn.turbo.config.enn_x_scaling import ENNMetricLearning, ENNScaleX

from .cat_ties import resolve_tied_groups
from .enn_class_support import _to_rust_seeds
from .mbpann import MBPANNMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    from enn.semantic import SemanticInputTransform

    from .enn_normal import ENNNormal
    from .enn_params import ENNParams, PosteriorFlags


class _EnnRustView:
    @property
    def _index_driver(self) -> ENNIndexDriver:
        return index_driver_from_wire(str(self._rust_model.index_driver))

    @property
    def tied_dims(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(i) for i in g) for g in self._rust_model.tied_groups())

    @property
    def scale_x(self) -> ENNScaleX:
        return ENNScaleX.ON if bool(self._rust_model.scale_x) else ENNScaleX.OFF

    @property
    def metric_learning(self) -> ENNMetricLearning:
        if bool(self._rust_model.metric_learning_auto):
            return ENNMetricLearning.AUTO
        return ENNMetricLearning.NONE

    @property
    def weighting(self) -> bool:
        return bool(self._rust_model.weighting)

    @property
    def metric(self) -> MBPANNMetric | None:
        if self.metric_learning != ENNMetricLearning.AUTO:
            return None
        return MBPANNMetric(self)


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
    draws_arr = np.asarray(draws, dtype=float)
    idx_arr = np.array(idx, dtype=int) if idx else np.zeros((x.shape[0], 0), dtype=int)
    return draws_arr, idx_arr


class EpistemicNearestNeighbors(_EnnRustView):
    input_transform: SemanticInputTransform | None = None
    semantic_only: bool = False
    _original_inputs = None

    @staticmethod
    def _coerce_inputs(train_x, train_y, train_yvar):
        train_x = np.asarray(train_x, dtype=float)
        train_y = np.asarray(train_y, dtype=float)
        if train_yvar is not None:
            train_yvar = np.asarray(train_yvar, dtype=float)
        return train_x, train_y, train_yvar

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_yvar: np.ndarray | None = None,
        tied_dims: Sequence[Sequence[int]] | None = None,
        input_transform: SemanticInputTransform | None = None,
        *,
        weighting: bool = False,
        index_driver: ENNIndexDriver = ENNIndexDriver.FLAT,
        work_dir: str | os.PathLike[str] | None = None,
        y_bounds: np.ndarray | None = None,
        semantic_only: bool = False,
    ) -> None:
        if input_transform is not None:
            self.input_transform = input_transform
        if semantic_only:
            self.semantic_only = True
        original_inputs = (
            np.asarray(train_x, dtype=float) if input_transform is not None else None
        )
        groups = [
            list(g)
            for g in resolve_tied_groups(
                train_x, tied_dims, semantic_only=semantic_only
            )
        ]
        if input_transform is not None:
            train_x = input_transform.transform(train_x, semantic_only=semantic_only)
        elif semantic_only:
            raise ValueError("semantic_only requires input_transform")
        validate_enn_placement(
            index_driver=index_driver,
            work_dir=work_dir,
            weighting=weighting,
        )
        train_x, train_y, train_yvar = self._coerce_inputs(train_x, train_y, train_yvar)
        if y_bounds is not None:
            y_bounds = np.asarray(y_bounds, dtype=float)
        rust_kwargs: dict[str, Any] = {
            "train_x": train_x,
            "train_y": train_y,
            "train_yvar": train_yvar,
            "weighting": weighting,
            "index_driver": index_driver_to_wire(index_driver),
            "tied_dims": groups,
        }
        if work_dir is not None:
            rust_kwargs["work_dir"] = os.fspath(work_dir)
        if y_bounds is not None:
            rust_kwargs["y_bounds"] = y_bounds
        self._rust_model = _RustENN(**rust_kwargs)
        if input_transform is not None:
            self._original_inputs = InputHistory(original_inputs, self._rust_model)

    def add(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yvar: np.ndarray | None = None,
    ) -> ENNAddToken:
        x, y, yvar = self._coerce_inputs(x, y, yvar)
        token = self._rust_model.add(self.transform_inputs(x), y, yvar)
        if self._original_inputs is not None:
            self._original_inputs.append(x)
        return token

    def transform_inputs(self, x):
        return (
            x
            if self.input_transform is None
            else self.input_transform.transform(x, semantic_only=self.semantic_only)
        )

    def ensure_index_sync(self) -> None:
        self._rust_model.ensure_index_sync()

    def schedule_background_flush(self) -> None:
        self._rust_model.schedule_background_flush()

    def persist_index_to_disk(self) -> None:
        self._rust_model.persist_index_to_disk()

    def train_rows_at(
        self, indices: list[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        idx = [int(i) for i in np.asarray(indices, dtype=int).ravel()]
        x, y, yvar = self._rust_model.train_rows_at(idx)
        x_arr = np.asarray(x, dtype=float)
        if self._original_inputs is not None:
            x_arr = self._original_inputs.gather(idx)
        y_arr = np.asarray(y, dtype=float)
        yvar_arr = None if yvar is None else np.asarray(yvar, dtype=float)
        return x_arr, y_arr, yvar_arr

    def index_memory_bytes(self) -> int:
        return int(self._rust_model.index_memory_bytes())

    @property
    def num_outputs(self) -> int:
        return int(self._rust_model.num_outputs)

    @property
    def rust_backend(self):
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
    def _train_y(self) -> np.ndarray:
        n = len(self)
        _, y, _ = self.train_rows_at(list(range(n)))
        return y

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

        mu, se, se_epi, se_ale, idx = self._rust_model.posterior(
            self.transform_inputs(x),
            k_num_neighbors=params.k_num_neighbors,
            epistemic_variance_scale=params.epistemic_variance_scale,
            aleatoric_variance_scale=params.aleatoric_variance_scale,
            exclude_nearest=flags.exclude_nearest,
            observation_noise=flags.observation_noise,
        )
        idx_arr = np.asarray(idx, dtype=int) if idx is not None else None
        yb = np.asarray(self._rust_model.y_bounds, dtype=float)
        return ENNNormal(mu, se, se_epi, se_ale, idx=idx_arr, y_bounds=yb)

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

        mu, se, se_epi, se_ale, idx = self._rust_model.conditional_posterior(
            self.transform_inputs(x_whatif),
            y_whatif,
            self.transform_inputs(x),
            k_num_neighbors=params.k_num_neighbors,
            epistemic_variance_scale=params.epistemic_variance_scale,
            aleatoric_variance_scale=params.aleatoric_variance_scale,
            exclude_nearest=flags.exclude_nearest,
            observation_noise=flags.observation_noise,
        )
        idx_arr = np.asarray(idx, dtype=int) if idx is not None else None
        yb = np.asarray(self._rust_model.y_bounds, dtype=float)
        return ENNNormal(mu, se, se_epi, se_ale, idx=idx_arr, y_bounds=yb)

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
        k_values = [p.k_num_neighbors for p in paramss]
        epistemic_scales = [p.epistemic_variance_scale for p in paramss]
        aleatoric_scales = [p.aleatoric_variance_scale for p in paramss]
        mu_all, se_all, se_epi_all, se_ale_all, idx = self._rust_model.batch_posterior(
            self.transform_inputs(x),
            k_values=k_values,
            epistemic_scales=epistemic_scales,
            aleatoric_scales=aleatoric_scales,
            exclude_nearest=flags.exclude_nearest,
            observation_noise=flags.observation_noise,
        )
        idx_arr = np.asarray(idx, dtype=int) if idx is not None else None
        yb = np.asarray(self._rust_model.y_bounds, dtype=float)
        return ENNNormal(
            mu_all, se_all, se_epi_all, se_ale_all, idx=idx_arr, y_bounds=yb
        )

    def neighbors(
        self, x: np.ndarray, k: int, *, exclude_nearest: bool = False
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        idx_2d = np.asarray(
            self._rust_model.neighbors(
                self.transform_inputs(x), int(k), exclude_nearest=exclude_nearest
            ),
            dtype=np.int64,
        )
        if idx_2d.size == 0:
            return np.zeros((0,), dtype=np.int64)
        return idx_2d[0, :]

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
        draws, idx = self._rust_model.posterior_function_draw(
            self.transform_inputs(x), **kw
        )
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
        seeds = _to_rust_seeds(function_seeds)
        kw = _rust_function_draw_kwargs(params, flags, seeds)
        draws, idx = self._rust_model.conditional_posterior_function_draw(
            self.transform_inputs(x_whatif),
            y_whatif,
            self.transform_inputs(x),
            **kw,
        )
        return _finalize_function_draw(x, draws, idx)
