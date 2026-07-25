from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


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
    # (a, b) logit inverse
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

        # Bound-aware: φ⁻¹(φ(μ_nat) + (se_nat / |dφ⁻¹/dz|) · ε)
        bounds = np.asarray(self.y_bounds, dtype=float)
        mu = np.asarray(self.mu, dtype=float)
        se = np.asarray(self.se, dtype=float)
        # metrics on last axis of mu/se
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
            # avoid divide-by-zero on flat regions
            scale = np.where(jac > 0, se_j / jac, 0.0)
            z_s = z_mu[..., None] + scale[..., None] * eps[..., j, :]
            out[..., j, :] = _inv_scalar(z_s, a, b)
        return out
