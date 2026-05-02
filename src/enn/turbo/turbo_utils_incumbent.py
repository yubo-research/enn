from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

__all__ = [
    "compute_full_box_bounds_1d",
    "get_single_incumbent_index",
    "get_incumbent_index",
    "get_scalar_incumbent_value",
    "ScalarIncumbentMixin",
]


def compute_full_box_bounds_1d(
    x_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lb = np.zeros_like(x_center, dtype=float)
    ub = np.ones_like(x_center, dtype=float)
    return lb, ub


def get_single_incumbent_index(
    selector: Any,
    y: np.ndarray,
    rng: Generator,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.array([], dtype=int)
    best_idx = selector.select(y, mu, rng)
    return np.array([best_idx])


def get_incumbent_index(
    selector: Any,
    y: np.ndarray,
    rng: Generator,
    mu: np.ndarray | None = None,
) -> int:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        raise ValueError("y is empty")
    return int(selector.select(y, mu, rng))


def get_scalar_incumbent_value(
    selector: Any,
    y_obs: np.ndarray,
    rng: Generator,
    *,
    mu_obs: np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(y_obs, dtype=float)
    if y.size == 0:
        return np.array([], dtype=float)
    idx = get_incumbent_index(selector, y, rng, mu=mu_obs)
    use_mu = bool(getattr(selector, "noise_aware", False))
    values = mu_obs if use_mu else y
    if values is None:
        raise ValueError("noise_aware incumbent selection requires mu_obs")
    v = np.asarray(values, dtype=float)
    if v.ndim == 2:
        value = float(v[idx, 0])
    elif v.ndim == 1:
        value = float(v[idx])
    else:
        raise ValueError(v.shape)
    return np.array([value], dtype=float)


class ScalarIncumbentMixin:
    incumbent_selector: Any

    def get_incumbent_index(
        self,
        y: np.ndarray | Any,
        rng: Generator,
        mu: np.ndarray | None = None,
    ) -> int:
        return get_incumbent_index(self.incumbent_selector, y, rng, mu=mu)

    def get_incumbent_value(
        self,
        y_obs: np.ndarray | Any,
        rng: Generator,
        mu_obs: np.ndarray | None = None,
    ) -> np.ndarray:
        return get_scalar_incumbent_value(
            self.incumbent_selector, y_obs, rng, mu_obs=mu_obs
        )
