from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator

from .. import _rust
from .config.acquisition import (
    DrawAcquisitionConfig,
    ParetoAcquisitionConfig,
    RandomAcquisitionConfig,
    UCBAcquisitionConfig,
)
from .config.candidate_gen_config import CandidateGenConfig
from .config.candidate_rv import CandidateRV
from .config.init_strategies import LHDOnlyInit
from .config.num_candidates_fn import default_num_candidates
from .config.optimizer_config import OptimizerConfig
from .config.surrogate import ENNSurrogateConfig, NoSurrogateConfig
from .config.trust_region import NoTRConfig, TurboTRConfig
from .optimizer import Optimizer as PythonOptimizer
from .types.telemetry import Telemetry

# Constants matching Python's default_num_candidates: min(5000, 100*num_dim)
_DEFAULT_NUM_CANDIDATES_FACTOR = 100.0
_DEFAULT_MAX_CANDIDATES = 5000


class _ObsView:
    """Minimal view wrapper for observation arrays (compat with Python optimizer).

    When empty, defaults to shape (0, 1). After tell() with multi-objective y,
    _y_obs reflects the actual shape (n, m) from the inner optimizer.
    """

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = np.asarray(arr, dtype=float)

    def view(self) -> np.ndarray:
        return self._arr


def _acquisition_to_override(config: OptimizerConfig) -> dict[str, Any]:
    """Map acquisition config to Rust override dict."""
    acq = getattr(config, "acquisition", None)
    if acq is None:
        return {}
    if isinstance(acq, UCBAcquisitionConfig):
        return {
            "acquisition": "ucb",
            "acquisition_beta": float(getattr(acq, "beta", 2.0)),
        }
    if isinstance(acq, DrawAcquisitionConfig):
        return {"acquisition": "thompson"}
    if isinstance(acq, RandomAcquisitionConfig):
        return {"acquisition": "random"}
    if isinstance(acq, ParetoAcquisitionConfig):
        return {"acquisition": "pareto"}
    return {}


def _can_use_rust_num_candidates(config: OptimizerConfig) -> bool:
    """True iff num_candidates matches Python default: min(5000, 100*num_dim).

    Rust uses a factor model; constant-in-dim or custom fns are not mappable.
    """
    candidates = getattr(config, "candidates", None)
    if not isinstance(candidates, CandidateGenConfig):
        return True  # No custom config; use Rust default
    fn = getattr(candidates, "num_candidates", None)
    if not callable(fn):
        return True
    if fn is default_num_candidates:
        return True
    # const_num_candidates or other custom: cannot map to Rust factor model
    if fn(num_dim=2, num_arms=1) == fn(num_dim=10, num_arms=1):
        return False  # Constant in dim -> const_num_candidates or similar
    # Non-default, non-constant: e.g. min(5000, 50*dim) - cannot infer safely
    return False


def _candidates_to_override(config: OptimizerConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rv = getattr(config, "candidate_rv", None)
    if rv is CandidateRV.SOBOL:
        out["candidate_rv"] = "sobol"
    elif rv is CandidateRV.UNIFORM:
        out["candidate_rv"] = "uniform"
    elif rv is CandidateRV.RAASP:
        out["candidate_rv"] = "raasp"
    candidates = getattr(config, "candidates", None)
    if isinstance(candidates, CandidateGenConfig):
        fn = getattr(candidates, "num_candidates", None)
        if fn is default_num_candidates:
            # Python default: min(5000, 100*num_dim). Map to Rust factor + cap.
            out["num_candidates_factor"] = _DEFAULT_NUM_CANDIDATES_FACTOR
            out["max_candidates"] = _DEFAULT_MAX_CANDIDATES
    return out


def _get_tr_params(tr: TurboTRConfig) -> tuple[float, float, float]:
    """Extract length_init, length_min, length_max from trust region config."""
    li = (
        tr.length_init
        if hasattr(tr, "length_init")
        else getattr(tr.length, "length_init", 0.8)
    )
    lm = (
        tr.length_min
        if hasattr(tr, "length_min")
        else getattr(tr.length, "length_min", 0.5**7)
    )
    lx = (
        tr.length_max
        if hasattr(tr, "length_max")
        else getattr(tr.length, "length_max", 1.6)
    )
    return float(li), float(lm), float(lx)


def _trust_region_to_override(config: OptimizerConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tr = getattr(config, "trust_region", None)
    if not isinstance(tr, TurboTRConfig):
        return out
    li, lm, lx = _get_tr_params(tr)
    if li != 0.8:
        out["length_init"] = li
    if abs(lm - 0.5**7) > 1e-12:
        out["length_min"] = lm
    if lx != 1.6:
        out["length_max"] = lx
    return out


def _config_to_rust_overrides(config: OptimizerConfig) -> dict[str, Any] | None:
    """Extract config overrides for Rust backend (config pass-through)."""
    overrides: dict[str, Any] = {}
    overrides.update(_acquisition_to_override(config))
    overrides.update(_candidates_to_override(config))
    overrides.update(_trust_region_to_override(config))
    surrogate = getattr(config, "surrogate", None)
    if isinstance(surrogate, ENNSurrogateConfig):
        from .config.enn_index_driver import ENN_INDEX_DRIVER_TO_RUST

        if surrogate.index_driver in ENN_INDEX_DRIVER_TO_RUST:
            overrides["index_driver"] = ENN_INDEX_DRIVER_TO_RUST[surrogate.index_driver]
        if surrogate.num_fit_samples is not None:
            overrides["num_fit_samples"] = int(surrogate.num_fit_samples)
        if surrogate.num_fit_candidates is not None:
            overrides["num_fit_candidates"] = int(surrogate.num_fit_candidates)
    trailing_obs = getattr(config, "trailing_obs", None)
    if trailing_obs is not None:
        overrides["trailing_obs"] = int(trailing_obs)
    return overrides if overrides else None


def is_rust_supported_config(config: OptimizerConfig) -> bool:
    """Check if the configuration can use the Rust optimizer."""
    if not _can_use_rust_num_candidates(config):
        return False  # const_num_candidates or other custom; fall back to Python
    if isinstance(config.surrogate, ENNSurrogateConfig):
        return config.surrogate.k is not None
    if isinstance(config.surrogate, NoSurrogateConfig):
        return True
    return False


def _is_lhd_only_config(config: OptimizerConfig) -> bool:
    """True if config is LHD_ONLY (NoTR + LHDOnlyInit + NoSurrogate)."""
    return (
        isinstance(config.trust_region, NoTRConfig)
        and isinstance(config.init.init_strategy, LHDOnlyInit)
        and isinstance(config.surrogate, NoSurrogateConfig)
    )


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

        # Convert from unit to original bounds
        lower = self._bounds[:, 0]
        upper = self._bounds[:, 1]
        return arms_unit * (upper - lower) + lower

    def tell(
        self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if x_arr.ndim != 2 or x_arr.shape[1] != self._num_dim:
            raise ValueError(
                f"x must have shape (n, {self._num_dim}), got {x_arr.shape}"
            )
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.ndim != 2 or y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(
                f"y must have shape ({x_arr.shape[0]}, m), got {y_arr.shape}"
            )

        # Convert x to unit space
        lower = self._bounds[:, 0]
        upper = self._bounds[:, 1]
        x_unit = (x_arr - lower) / (upper - lower)

        seed = int(self._rng.integers(2**63 - 1))
        self._inner.tell(x_unit, y_arr, seed)

        return y_arr


def create_optimizer(
    *,
    bounds: np.ndarray,
    config: OptimizerConfig,
    rng: Generator,
) -> RustOptimizer | PythonOptimizer:
    """Create optimizer, using Rust backend when possible."""
    if not is_rust_supported_config(config):
        from .optimizer import create_optimizer as create_python_optimizer

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
        k = config.surrogate.k
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
