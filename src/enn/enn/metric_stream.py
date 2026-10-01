"""Thin wrappers around the Rust AUTO metric estimators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from enn import _rust

MIN_DEPENDENCE_ROWS = 100


def _y2(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def _seed_of(seed) -> int:
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    raise TypeError("Reservoir seed must be an int")


class Reservoir:
    """Uniform sample of up to ``capacity`` rows. ``seed`` is a NumPy PCG64 seed."""

    def __init__(
        self, capacity: int, num_dim: int, seed: int, num_outputs: int = 1
    ) -> None:
        self._inner = _rust.PyReservoir(
            int(capacity), int(num_dim), _seed_of(seed), int(num_outputs)
        )

    @property
    def capacity(self) -> int:
        return int(self._inner.capacity)

    @property
    def num_seen(self) -> int:
        return int(self._inner.num_seen)

    def __len__(self) -> int:
        return int(self._inner.__len__())

    @property
    def x(self) -> np.ndarray:
        return np.asarray(self._inner.x(), dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self._inner.y(), dtype=float)

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = _y2(y)
        self._inner.add(x, y)


def sobol_index(
    x: np.ndarray, y: np.ndarray, num_bins: int | None = None
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    bins = None if num_bins is None else int(num_bins)
    return np.asarray(_rust.sobol_index(x, y, bins), dtype=float)


def group_sobol_index(x_group: np.ndarray, y: np.ndarray) -> float:
    return float(
        _rust.group_sobol_index(
            np.asarray(x_group, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        )
    )


def null_sd(n: int, num_cells: int | None = None) -> float:
    return float(_rust.null_sd(int(n), None if num_cells is None else int(num_cells)))


def dependence_weights(
    x: np.ndarray,
    y: np.ndarray,
    floor: float | None = None,
    tied: Sequence[Sequence[int]] = (),
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = _y2(y)
    groups = [list(map(int, g)) for g in tied]
    used = None if floor is None else float(floor)
    return np.asarray(_rust.dependence_weights(x, y, groups, used), dtype=float)


def loo_loglik(x: np.ndarray, y: np.ndarray, a: np.ndarray, k: int) -> float:
    return float(
        _rust.loo_loglik(
            np.asarray(x, dtype=float), _y2(y), np.asarray(a, dtype=float), int(k)
        )
    )


def auto_weights(
    x: np.ndarray, y: np.ndarray, k: int, tied: Sequence[Sequence[int]] = ()
) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    y = _y2(y)
    groups = [list(map(int, g)) for g in tied]
    w, gain = _rust.auto_weights(x, y, int(k), groups)
    return np.asarray(w, dtype=float), float(gain)
