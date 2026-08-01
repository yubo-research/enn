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
    resolve_enn_k,
)
from .types.telemetry import Telemetry


class _ObsView:
    """Minimal view wrapper for observation arrays (compat with Python optimizer).

    When empty, defaults to shape (0, 1). After tell() with multi-objective y,
    _y_obs reflects the actual shape (n, m) from the inner optimizer.
    """

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = np.asarray(arr, dtype=float)

    def view(self) -> np.ndarray:
        return self._arr


class RustOptimizer:
    """Python wrapper around the Rust-backed optimizer."""

    def __init__(
        self,
        bounds: np.ndarray,
        config: OptimizerConfig,
        rng: Generator,
        inner: Any,
    ) -> None:
        self._bounds = np.asarray(bounds, dtype=float)
        self._config = config
        self._rng = rng
        self._inner = inner
        self._num_dim = self._bounds.shape[0]
        self._expects_yvar: bool | None = None

    @property
    def _x_obs(self) -> _ObsView:
        fn = getattr(self._inner, "x_obs", None)
        if fn is None or not callable(fn):
            return _ObsView(np.empty((0, self._num_dim)))
        arr = fn()
        return _ObsView(
            np.empty((0, self._num_dim)) if arr is None else np.asarray(arr)
        )

    @property
    def _y_obs(self) -> _ObsView:
        fn = getattr(self._inner, "y_obs", None)
        if fn is None or not callable(fn):
            return _ObsView(np.empty((0, 1)))
        arr = fn()
        return _ObsView(np.empty((0, 1)) if arr is None else np.asarray(arr))

    @property
    def tr_obs_count(self) -> int:
        tr_obs_count = getattr(self._inner, "tr_obs_count", None)
        if callable(tr_obs_count):
            return int(tr_obs_count())
        if tr_obs_count is not None:
            return int(tr_obs_count)
        return 0

    @property
    def tr_length(self) -> float:
        tr_length = getattr(self._inner, "tr_length", None)
        if callable(tr_length):
            return float(tr_length())
        if tr_length is not None:
            return float(tr_length)
        return 0.5

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
        num_arms = int(num_arms)
        if num_arms <= 0:
            raise ValueError(f"num_arms must be > 0, got {num_arms}")

        seed = int(self._rng.integers(2**63 - 1))
        arms_unit = self._inner.ask(num_arms, seed)

        lower = self._bounds[:, 0]
        upper = self._bounds[:, 1]
        return arms_unit * (upper - lower) + lower

    def tell(
        self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None
    ) -> np.ndarray:
        from .python_fallback.turbo_optimizer_utils import validate_tell_inputs

        inputs = validate_tell_inputs(x, y, y_var, self._num_dim)
        if self._expects_yvar is None:
            self._expects_yvar = inputs.y_var is not None
        if (inputs.y_var is not None) != bool(self._expects_yvar):
            raise ValueError(
                f"y_var must be {'provided' if self._expects_yvar else 'omitted'} on every tell()"
            )
        if inputs.x.shape[0] == 0:
            return (
                np.array([], dtype=float)
                if inputs.num_metrics == 1
                else np.empty((0, inputs.num_metrics), dtype=float)
            )

        lower = self._bounds[:, 0]
        upper = self._bounds[:, 1]
        x_unit = (inputs.x - lower) / (upper - lower)

        seed = int(self._rng.integers(2**63 - 1))
        if inputs.y_var is None:
            self._inner.tell(x_unit, inputs.y, seed)
        else:
            self._inner.tell(x_unit, inputs.y, seed, inputs.y_var)

        return inputs.y


def create_optimizer(
    *,
    bounds: np.ndarray,
    config: OptimizerConfig,
    rng: Generator,
) -> Any:
    """Create optimizer, using Rust backend when possible."""
    if not is_rust_supported_config(config):
        from .python_fallback.optimizer import (
            create_optimizer as create_python_optimizer,
        )

        return create_python_optimizer(bounds=bounds, config=config, rng=rng)

    bounds_arr = np.asarray(bounds, dtype=float)
    seed = int(rng.integers(2**63 - 1))
    num_init = config.init.num_init
    n_init = num_init if num_init is not None else 10
    overrides = _config_to_rust_overrides(config)

    if _is_lhd_only_config(config):
        inner = _rust.create_optimizer_lhd(
            bounds_arr, n_init, seed, config_overrides=overrides
        )
    elif isinstance(config.surrogate, ENNSurrogateConfig):
        k = resolve_enn_k(config)
        inner = _rust.create_optimizer_enn(
            bounds_arr, k, n_init, seed, config_overrides=overrides
        )
    elif isinstance(config.surrogate, NoSurrogateConfig):
        inner = _rust.create_optimizer_zero(
            bounds_arr, n_init, seed, config_overrides=overrides
        )
    else:
        raise ValueError(f"Unsupported surrogate config: {type(config.surrogate)}")

    return RustOptimizer(bounds=bounds, config=config, rng=rng, inner=inner)
