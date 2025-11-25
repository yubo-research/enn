from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np


@dataclass
class GumbelTrustRegion:
    num_dim: int
    length: float = 1.0
    neighbors_fn: (
        Callable[[np.ndarray, int], list[tuple[np.ndarray, np.ndarray]]] | None
    ) = None

    def update(
        self,
        values: np.ndarray | Any,
        x_center: np.ndarray | Any | None = None,
        k: int | None = None,
    ) -> None:
        import numpy as np

        from .enn_util import gumbel_expected_max

        if (
            self.neighbors_fn is not None
            and x_center is not None
            and k is not None
            and k > 0
        ):
            neighbors = self.neighbors_fn(x_center, k)
            if len(neighbors) == 0:
                self.length = 1.0
                return
            y_values = np.array([y for _, y in neighbors])
        else:
            y_values = values

        n = len(y_values)
        if n <= 1:
            self.length = 1.0
            return
        y_max = float(np.max(y_values))
        y_median = float(np.median(y_values))
        y_std = float(np.std(y_values))
        denom = 2.0 * gumbel_expected_max(n)
        if denom <= 0:
            denom = 1.0
        signal = ((y_max - y_median) / (1e-6 + y_std) / denom) ** 2
        scale = 1.0 / (1e-6 + signal)
        self.length = float(np.clip(scale, 0.1, 1.0))

    def needs_restart(self) -> bool:
        return False

    def restart(self) -> None:
        pass

    def compute_bounds_1d(
        self, x_center: np.ndarray | Any, weights: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        length = self.length
        if weights is None:
            half_length = 0.5 * length
        else:
            half_length = weights * length / 2.0
        lb = np.clip(x_center - half_length, 0.0, 1.0)
        ub = np.clip(x_center + half_length, 0.0, 1.0)
        return lb, ub
