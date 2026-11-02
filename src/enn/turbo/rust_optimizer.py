from __future__ import annotations

import os
from typing import Any

import numpy as np

from .. import _rust
from .config.enn_surrogate_config import validate_enn_placement
from .config.optimizer_config import OptimizerConfig
from .config.surrogate import ENNSurrogateConfig, NoSurrogateConfig
from .rust_optimizer_helpers import (
    _config_to_rust_overrides,
    _is_lhd_only_config,
    is_rust_supported_config,
)
from .types.telemetry import Telemetry


def _tell_estimate(inner: Any, train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    if not inner.has_surrogate():
        return train_y
    return np.asarray(inner.posterior_mu(train_x), dtype=float)


class Optimizer:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @classmethod
    def open(cls, work_dir: str | os.PathLike[str]) -> Optimizer:
        return cls(_rust.open_optimizer(os.fspath(work_dir)))

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(self._inner.bounds(), dtype=float)

    @property
    def x_obs(self) -> np.ndarray:
        arr = self._inner.x_obs()
        if arr is None:
            return np.empty((0, self.bounds.shape[0]))
        return np.asarray(arr, dtype=float)

    @property
    def y_obs(self) -> np.ndarray:
        arr = self._inner.y_obs()
        if arr is None:
            return np.empty((0, 1))
        return np.asarray(arr, dtype=float)

    @property
    def incumbent_x(self) -> np.ndarray | None:
        arr = self._inner.incumbent_x()
        return None if arr is None else np.asarray(arr, dtype=float)

    @property
    def best_x(self) -> np.ndarray | None:
        arr = self._inner.best_x()
        return None if arr is None else np.asarray(arr, dtype=float)

    @property
    def best_y(self) -> np.ndarray | None:
        arr = self._inner.best_y()
        return None if arr is None else np.asarray(arr, dtype=float)

    @property
    def tr_obs_count(self) -> int:
        return int(self._inner.tr_obs_count())

    @property
    def tr_length(self) -> float:
        return float(self._inner.tr_length())

    def telemetry(self) -> Telemetry:
        t = self._inner.telemetry()
        return Telemetry(
            dt_fit=t.dt_fit,
            dt_gen=t.dt_gen,
            dt_sel=t.dt_sel,
            dt_tell=t.dt_tell,
            num_candidates=int(t.num_candidates),
        )

    @property
    def init_progress(self) -> tuple[int, int] | None:
        result = self._inner.init_progress()
        if result is None:
            return None
        return result

    def ask(self, num_arms: int) -> np.ndarray:
        return np.asarray(self._inner.ask(int(num_arms)), dtype=float)

    def tell(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_yvar: np.ndarray | None = None,
    ) -> np.ndarray:
        train_x = np.asarray(train_x, dtype=float)
        train_y = np.asarray(train_y, dtype=float)
        if train_yvar is not None:
            train_yvar = np.asarray(train_yvar, dtype=float)
        self._inner.tell(train_x, train_y, train_yvar)
        if train_y.shape[0] == 0:
            return train_y
        return _tell_estimate(self._inner, train_x, train_y)


def create_optimizer(
    *,
    bounds: np.ndarray,
    config: OptimizerConfig,
    seed: int,
    work_dir: str | os.PathLike[str],
) -> Optimizer:
    if not is_rust_supported_config(config):
        raise ValueError(f"Unsupported optimizer config: {type(config.surrogate)}")

    bounds_arr = np.asarray(bounds, dtype=float)
    transform = getattr(config.surrogate, "input_transform", None)
    if transform is not None and not np.array_equal(transform.bounds, bounds_arr):
        raise ValueError("semantic transform bounds must match optimizer bounds")
    seed = int(seed)
    work_dir = os.fspath(work_dir)
    validate_enn_placement(work_dir)
    num_init = config.init.num_init
    overrides = _config_to_rust_overrides(config)

    if _is_lhd_only_config(config):
        kind, k = "lhd", None
    elif isinstance(config.surrogate, ENNSurrogateConfig):
        kind = "enn"
        k = None if config.surrogate.k is None else int(config.surrogate.k)
    elif isinstance(config.surrogate, NoSurrogateConfig):
        kind, k = "zero", None
    else:
        raise ValueError(f"Unsupported surrogate config: {type(config.surrogate)}")

    inner = _rust.create_optimizer_run(
        kind,
        bounds_arr,
        work_dir,
        k,
        num_init,
        seed,
        config_overrides=overrides,
    )
    return Optimizer(inner)
