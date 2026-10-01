from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


def _z_crit(level: float) -> float:
    from enn._rust import z_crit

    return float(z_crit(float(level)))


@dataclass
class ENNNormal:
    mu: np.ndarray
    se: np.ndarray
    se_epi: np.ndarray
    se_ale: np.ndarray
    idx: np.ndarray | None = None
    y_bounds: np.ndarray | None = None

    def sample(
        self,
        num_samples: int,
        rng: Generator,
        clip: float | None = None,
    ) -> np.ndarray:
        import numpy as np

        from enn._rust import sample_normal

        seed = int(rng.integers(0, 2**63 - 1))
        mu = np.asarray(self.mu, dtype=float)
        se = np.asarray(self.se, dtype=float)
        mu2 = mu if mu.ndim == 2 else mu.reshape(1, -1)
        se2 = se if se.ndim == 2 else se.reshape(1, -1)
        bounds = None if self.y_bounds is None else np.asarray(self.y_bounds, dtype=float)
        flat = np.asarray(
            sample_normal(mu2, se2, int(num_samples), seed, bounds, clip), dtype=float
        )
        return flat.reshape(*se.shape, int(num_samples))

    def confidence_interval(
        self,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) Gaussian intervals with the same warp as sample()."""
        import numpy as np

        from enn._rust import confidence_interval

        mu = np.asarray(self.mu, dtype=float)
        se = np.asarray(self.se, dtype=float)
        mu2 = mu if mu.ndim == 2 else mu.reshape(1, -1)
        se2 = se if se.ndim == 2 else se.reshape(1, -1)
        bounds = None if self.y_bounds is None else np.asarray(self.y_bounds, dtype=float)
        lo, hi = confidence_interval(mu2, se2, float(level), bounds)
        return np.asarray(lo).reshape(mu.shape), np.asarray(hi).reshape(mu.shape)
