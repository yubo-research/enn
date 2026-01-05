from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


class TrustRegion(Protocol):
    @property
    def length(self) -> float: ...

    def compute_bounds(
        self, x_center: np.ndarray, lengthscales: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def update(self, y_values: np.ndarray) -> None: ...

    def needs_restart(self) -> bool: ...

    def restart(self) -> None: ...

    def get_incumbent_indices(self, y: np.ndarray, rng: Generator) -> np.ndarray: ...
