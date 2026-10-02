from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator

from .. import _rust
from .config.optimizer_config import OptimizerConfig
from .config.surrogate import ENNSurrogateConfig, NoSurrogateConfig
from .rust_optimizer_helpers import (
    _config_to_rust_overrides,
    _is_lhd_only_config,
    is_rust_supported_config,
)
from .types.telemetry import Telemetry


def _tell_estimate(inner: Any, x_native: np.ndarray, y_in: np.ndarray) -> np.ndarray:
    """Posterior mean at the points just told. Raw y if there is no surrogate."""
    x = np.asarray(x_native, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    try:
        mu = np.asarray(inner.posterior_mu(x), dtype=float)
    except ValueError as exc:
        if "No surrogate" not in str(exc):
            raise
        return y_in
    if y_in.ndim == 1 and mu.ndim == 2 and mu.shape[1] == 1:
        return mu.reshape(-1)
    return mu


class RustOptimizer:
    """Facade over the Rust optimizer. The only field is the Rust object."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def _dim(self) -> int:
        return int(np.asarray(self._inner.bounds()).shape[0])

    @property
    def _x_obs(self) -> np.ndarray:
        arr = self._inner.x_obs()
        if arr is None:
            return np.empty((0, self._dim()))
        return np.asarray(arr, dtype=float)

    @property
    def _y_obs(self) -> np.ndarray:
        arr = self._inner.y_obs()
        if arr is None:
            return np.empty((0, 1))
        return np.asarray(arr, dtype=float)

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
        self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None
    ) -> np.ndarray:
        x_native = np.asarray(x, dtype=float)
        y_in = np.asarray(y, dtype=float)
        y_native = y_in.reshape(-1, 1) if y_in.ndim == 1 else y_in
        if y_native.shape[0] == 0:
            return y_in
        if y_var is None:
            self._inner.tell(x_native, y_native)
        else:
            y_var_native = np.asarray(y_var, dtype=float)
            if y_var_native.ndim == 1:
                y_var_native = y_var_native.reshape(-1, 1)
            self._inner.tell(x_native, y_native, y_var_native)
        return _tell_estimate(self._inner, x_native, y_in)


def create_optimizer(
    *,
    bounds: np.ndarray,
    config: OptimizerConfig,
    rng: Generator,
) -> Any:
    """Create a Rust optimizer. `rng` is used only to draw the construction seed."""
    if not is_rust_supported_config(config):
        raise ValueError(f"Unsupported optimizer config: {type(config.surrogate)}")

    bounds_arr = np.asarray(bounds, dtype=float)
    seed = int(rng.integers(2**63 - 1))
    num_init = config.init.num_init
    overrides = _config_to_rust_overrides(config)

    if _is_lhd_only_config(config):
        inner = _rust.create_optimizer_lhd(
            bounds_arr, num_init, seed, config_overrides=overrides
        )
    elif isinstance(config.surrogate, ENNSurrogateConfig):
        k = None if config.surrogate.k is None else int(config.surrogate.k)
        inner = _rust.create_optimizer_enn(
            bounds_arr, k, num_init, seed, config_overrides=overrides
        )
    elif isinstance(config.surrogate, NoSurrogateConfig):
        inner = _rust.create_optimizer_zero(
            bounds_arr, num_init, seed, config_overrides=overrides
        )
    else:
        raise ValueError(f"Unsupported surrogate config: {type(config.surrogate)}")

    return RustOptimizer(inner)
