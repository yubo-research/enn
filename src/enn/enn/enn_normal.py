from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from enn._rust import ENNNormal as _RustENNNormal

if TYPE_CHECKING:
    pass


class ENNNormal:
    """Facade over the Rust posterior. The only field is that object."""

    def __init__(
        self,
        mu: np.ndarray,
        se: np.ndarray,
        se_epi: np.ndarray,
        se_ale: np.ndarray,
        idx: np.ndarray | None = None,
        y_bounds: np.ndarray | None = None,
    ) -> None:
        mu_arr = np.asarray(mu, dtype=float)
        se_arr = np.asarray(se, dtype=float)
        se_epi_arr = np.asarray(se_epi, dtype=float)
        se_ale_arr = np.asarray(se_ale, dtype=float)
        idx_arr = None if idx is None else np.asarray(idx)
        bounds = None if y_bounds is None else np.asarray(y_bounds, dtype=float)
        self._inner = _RustENNNormal(mu_arr, se_arr, se_epi_arr, se_ale_arr, idx_arr, bounds)

    @property
    def mu(self) -> np.ndarray:
        return np.asarray(self._inner.mu, dtype=float)

    @property
    def se(self) -> np.ndarray:
        return np.asarray(self._inner.se, dtype=float)

    @property
    def se_epi(self) -> np.ndarray:
        return np.asarray(self._inner.se_epi, dtype=float)

    @property
    def se_ale(self) -> np.ndarray:
        return np.asarray(self._inner.se_ale, dtype=float)

    @property
    def idx(self) -> np.ndarray | None:
        idx = self._inner.idx
        if idx is None:
            return None
        return np.asarray(idx)

    @property
    def y_bounds(self) -> np.ndarray | None:
        bounds = self._inner.y_bounds
        if bounds is None:
            return None
        return np.asarray(bounds, dtype=float)

    def sample(
        self,
        num_samples: int,
        seed: int,
        clip: float | None = None,
    ) -> np.ndarray:
        return np.asarray(
            self._inner.sample(int(num_samples), int(seed), clip),
            dtype=float,
        )

    def confidence_interval(
        self,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self._inner.confidence_interval(float(level))
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
