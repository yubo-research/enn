from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from enn._rust import (
    arms_from_pareto_fronts as _rust_arms_from_pareto_fronts,
)
from enn._rust import (
    calculate_sobol_indices as _rust_calculate_sobol_indices,
)
from enn._rust import (
    pareto_front_2d_maximize as _rust_pareto_front_2d_maximize,
)
from enn._rust import (
    standardize_y as _rust_standardize_y,
)

if TYPE_CHECKING:
    from numpy.random import Generator


def standardize_y(y: np.ndarray | list[float] | Any) -> tuple[float, float]:
    y_array = np.asarray(y, dtype=float)
    center, scale = _rust_standardize_y(y_array)
    return float(center), float(scale)


def calculate_sobol_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate Sobol indices using Rust backend."""
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    n, d = x.shape
    if d <= 0:
        raise ValueError(f"x must have at least 1 dimension, got {d}")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(f"y shape {y.shape} incompatible with x rows {n}")

    x_f64 = np.asarray(x, dtype=np.float64)
    y_f64 = np.asarray(y, dtype=np.float64)
    result = _rust_calculate_sobol_indices(x_f64, y_f64)
    return np.asarray(result, dtype=x.dtype)


def pareto_front_2d_maximize(
    a: np.ndarray | Any, b: np.ndarray | Any, idx: np.ndarray | Any | None = None
) -> np.ndarray:
    """Compute 2D Pareto front (maximize both objectives) using Rust backend."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError((a.shape, b.shape))
    if idx is None:
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("a and b must be finite")
        return np.asarray(_rust_pareto_front_2d_maximize(a, b), dtype=int)
    idx_arr = np.asarray(idx, dtype=int)
    if idx_arr.ndim != 1:
        raise ValueError(idx_arr.shape)
    n = a.size
    for i in idx_arr:
        if i < 0:
            raise ValueError(f"idx entry {i} is negative")
        if i >= n:
            raise ValueError(f"idx entry {i} is out of bounds for length {n}")
        if not np.isfinite(a[i]) or not np.isfinite(b[i]):
            raise ValueError("a and b must be finite")
    return np.asarray(_rust_pareto_front_2d_maximize(a, b, idx_arr), dtype=int)


def arms_from_pareto_fronts(
    x_cand: np.ndarray | Any,
    mu: np.ndarray | Any,
    se: np.ndarray | Any,
    num_arms: int,
    rng: Generator | Any,
) -> np.ndarray:
    x_array = np.asarray(x_cand, dtype=np.float64)
    mu_array = np.asarray(mu, dtype=np.float64)
    se_array = np.asarray(se, dtype=np.float64)
    if x_array.ndim != 2:
        raise ValueError(x_array.shape)
    if mu_array.shape != se_array.shape or mu_array.ndim != 1:
        raise ValueError((mu_array.shape, se_array.shape))
    if mu_array.size != x_array.shape[0]:
        raise ValueError((mu_array.size, x_array.shape[0]))
    num_arms = int(num_arms)
    if num_arms <= 0:
        raise ValueError(num_arms)
    if not np.all(np.isfinite(mu_array)) or not np.all(np.isfinite(se_array)):
        raise ValueError("mu and se must be finite")
    seed = int(rng.integers(0, 2**63 - 1))
    result = _rust_arms_from_pareto_fronts(x_array, mu_array, se_array, num_arms, seed)
    return np.asarray(result, dtype=x_array.dtype)
