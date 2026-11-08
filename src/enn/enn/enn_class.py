from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from enn._rust import EpistemicNearestNeighbors as _RustENN
from enn.turbo.config.enn_surrogate_config import validate_enn_placement
from enn.turbo.config.enn_x_scaling import ENNMetricLearning

from .cat_ties import resolve_tied_groups
from .enn_class_support import _to_rust_seeds, rust_params
from .mbpann import MBPANNMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .enn_normal import ENNNormal
    from .enn_params import ENNParams, PosteriorFlags


class _EnnRustView:
    @property
    def tied_dims(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(i) for i in g) for g in self._rust_model.tied_groups())

    @property
    def metric_learning(self) -> ENNMetricLearning:
        if bool(self._rust_model.metric_learning_auto):
            return ENNMetricLearning.AUTO
        return ENNMetricLearning.NONE

    @property
    def weighting(self) -> bool:
        return bool(self._rust_model.weighting)

    @property
    def aggregate_duplicates(self) -> bool:
        return bool(self._rust_model.aggregate_duplicates)

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
        "params": rust_params(params),
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
        *,
        work_dir: str | os.PathLike[str],
        weighting: bool = False,
        y_bounds: np.ndarray | None = None,
        aggregate_duplicates: bool = False,
    ) -> None:
        groups = [
            list(g)
            for g in resolve_tied_groups(train_x, tied_dims, weighting=weighting)
        ]
        validate_enn_placement(work_dir)
        train_x, train_y, train_yvar = self._coerce_inputs(train_x, train_y, train_yvar)
        if y_bounds is not None:
            y_bounds = np.asarray(y_bounds, dtype=float)
        rust_kwargs: dict[str, Any] = {
            "train_x": train_x,
            "train_y": train_y,
            "work_dir": os.fspath(work_dir),
            "train_yvar": train_yvar,
            "weighting": weighting,
            "tied_dims": groups,
            "aggregate_duplicates": aggregate_duplicates,
        }
        if y_bounds is not None:
            rust_kwargs["y_bounds"] = y_bounds
        self._rust_model = _RustENN(**rust_kwargs)

    @classmethod
    def open(cls, work_dir: str | os.PathLike[str]) -> EpistemicNearestNeighbors:
        validate_enn_placement(work_dir)
        self = cls.__new__(cls)
        self._rust_model = _RustENN.open(os.fspath(work_dir))
        return self

    def add(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_yvar: np.ndarray | None = None,
    ) -> None:
        train_x, train_y, train_yvar = self._coerce_inputs(train_x, train_y, train_yvar)
        self._rust_model.add(train_x, train_y, train_yvar)

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
            np.asarray(x, dtype=float),
            params=rust_params(params),
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
            np.asarray(x_whatif, dtype=float),
            np.asarray(y_whatif, dtype=float),
            np.asarray(x, dtype=float),
            params=rust_params(params),
            exclude_nearest=flags.exclude_nearest,
            observation_noise=flags.observation_noise,
        )
        idx_arr = np.asarray(idx, dtype=int) if idx is not None else None
        yb = np.asarray(self._rust_model.y_bounds, dtype=float)
        return ENNNormal(mu, se, se_epi, se_ale, idx=idx_arr, y_bounds=yb)

    def batch_posterior(
        self,
        x: np.ndarray,
        *,
        paramss: list[ENNParams],
        flags: PosteriorFlags | None = None,
    ) -> ENNNormal:
        from .enn_normal import ENNNormal

        flags = _posterior_flags_coerced(flags)
        x = np.asarray(x, dtype=float)
        mu_all, se_all, se_epi_all, se_ale_all, idx = self._rust_model.batch_posterior(
            x,
            paramss=[rust_params(p) for p in paramss],
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
        return np.asarray(
            self._rust_model.neighbors(x, int(k), exclude_nearest=exclude_nearest),
            dtype=np.int64,
        )

    def posterior_function_draw(
        self,
        x: np.ndarray,
        *,
        params: ENNParams,
        function_seeds: np.ndarray | list[int],
        flags: PosteriorFlags | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        flags = _posterior_flags_coerced(flags)
        x = np.asarray(x, dtype=float)
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
        x = np.asarray(x, dtype=float)
        seeds = _to_rust_seeds(function_seeds)
        kw = _rust_function_draw_kwargs(params, flags, seeds)
        draws, idx = self._rust_model.conditional_posterior_function_draw(
            np.asarray(x_whatif, dtype=float),
            np.asarray(y_whatif, dtype=float),
            x,
            **kw,
        )
        return _finalize_function_draw(x, draws, idx)
