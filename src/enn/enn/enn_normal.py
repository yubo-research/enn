from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


def _z_crit(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    target = 0.5 * (1.0 + level)
    lo, hi = 0.0, 8.0
    while 0.5 * (1.0 + math.erf(hi / math.sqrt(2.0))) < target:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _is_identity_bounds(y_bounds: np.ndarray | None) -> bool:
    import numpy as np

    if y_bounds is None:
        return True
    b = np.asarray(y_bounds, dtype=float)
    if b.size == 0:
        return True
    return bool(np.all(np.isneginf(b[:, 0]) & np.isposinf(b[:, 1])))


def _inv_scalar(z: np.ndarray, a: float, b: float) -> np.ndarray:
    import numpy as np

    if np.isneginf(a) and np.isposinf(b):
        return z
    if np.isfinite(a) and np.isposinf(b):
        return a + np.exp(z)
    if np.isneginf(a) and np.isfinite(b):
        return b - np.exp(-z)

    s = 1.0 / (1.0 + np.exp(-z))
    return a + (b - a) * s


def _d_inv_dz(z: np.ndarray, a: float, b: float) -> np.ndarray:
    import numpy as np

    if np.isneginf(a) and np.isposinf(b):
        return np.ones_like(z)
    if np.isfinite(a) and np.isposinf(b):
        return np.exp(z)
    if np.isneginf(a) and np.isfinite(b):
        return np.exp(-z)
    s = 1.0 / (1.0 + np.exp(-z))
    return (b - a) * s * (1.0 - s)


def _warp_scalar(y: np.ndarray, a: float, b: float) -> np.ndarray:
    import numpy as np

    if np.isneginf(a) and np.isposinf(b):
        return y
    if np.isfinite(a) and np.isposinf(b):
        return np.log(y - a)
    if np.isneginf(a) and np.isfinite(b):
        return -np.log(b - y)
    u = (y - a) / (b - a)
    return np.log(u / (1.0 - u))


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

        size = (*self.se.shape, num_samples)
        eps = rng.normal(size=size)
        if clip is not None:
            eps = np.clip(eps, a_min=-clip, a_max=clip)

        if _is_identity_bounds(self.y_bounds):
            return np.expand_dims(self.mu, -1) + np.expand_dims(self.se, -1) * eps


        bounds = np.asarray(self.y_bounds, dtype=float)
        mu = np.asarray(self.mu, dtype=float)
        se = np.asarray(self.se, dtype=float)

        m = bounds.shape[0]
        if mu.shape[-1] != m:
            raise ValueError(
                f"ENNNormal.mu last axis {mu.shape[-1]} != y_bounds rows {m}"
            )
        out = np.empty(size, dtype=float)
        for j in range(m):
            a, b = float(bounds[j, 0]), float(bounds[j, 1])
            mu_j = mu[..., j]
            se_j = se[..., j]
            z_mu = _warp_scalar(mu_j, a, b)
            jac = np.abs(_d_inv_dz(z_mu, a, b))

            scale = np.where(jac > 0, se_j / jac, 0.0)
            z_s = z_mu[..., None] + scale[..., None] * eps[..., j, :]
            out[..., j, :] = _inv_scalar(z_s, a, b)
        return out

    def confidence_interval(
        self,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) Gaussian intervals with the same warp as sample()."""
        import numpy as np

        z = _z_crit(level)
        mu = np.asarray(self.mu, dtype=float)
        se = np.asarray(self.se, dtype=float)

        if _is_identity_bounds(self.y_bounds):
            return mu - z * se, mu + z * se

        bounds = np.asarray(self.y_bounds, dtype=float)
        m = bounds.shape[0]
        if mu.shape[-1] != m:
            raise ValueError(
                f"ENNNormal.mu last axis {mu.shape[-1]} != y_bounds rows {m}"
            )
        lower = np.empty_like(mu)
        upper = np.empty_like(mu)
        for j in range(m):
            a, b = float(bounds[j, 0]), float(bounds[j, 1])
            mu_j = mu[..., j]
            se_j = se[..., j]
            z_mu = _warp_scalar(mu_j, a, b)
            jac = np.abs(_d_inv_dz(z_mu, a, b))
            scale = np.where(jac > 0, se_j / jac, 0.0)
            lower[..., j] = _inv_scalar(z_mu - z * scale, a, b)
            upper[..., j] = _inv_scalar(z_mu + z * scale, a, b)
        return lower, upper
