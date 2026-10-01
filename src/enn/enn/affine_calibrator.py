from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .enn_normal import ENNNormal

def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {out.shape}")
    return out


def _bounds(y_bounds: np.ndarray | None) -> np.ndarray | None:
    if y_bounds is None:
        return None
    return np.asarray(y_bounds, dtype=float)


def _align(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    while out.ndim < len(shape):
        out = np.expand_dims(out, axis=-2)
    return np.broadcast_to(out, shape)


@dataclass
class AffineCalibrator:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    @classmethod
    def identity(cls, num_metrics: int = 1) -> AffineCalibrator:
        m = int(num_metrics)
        return cls(
            a=np.zeros(m, dtype=float),
            b=np.ones(m, dtype=float),
            c=np.ones(m, dtype=float),
        )

    @classmethod
    def fit(
        cls,
        mu: np.ndarray,
        y: np.ndarray,
        se: np.ndarray | None = None,
        *,
        fit_residual_scale: bool = True,
    ) -> AffineCalibrator:
        mu2 = _as_2d(mu, "mu")
        y2 = _as_2d(y, "y")
        if mu2.shape != y2.shape:
            raise ValueError(f"mu shape {mu2.shape} != y shape {y2.shape}")
        from enn._rust import fit_affine

        se_arr = None if se is None else _as_2d(se, "se")
        a, b, c = fit_affine(mu2, y2, se_arr if fit_residual_scale else None)
        return cls(a=np.asarray(a, dtype=float), b=np.asarray(b, dtype=float), c=np.asarray(c, dtype=float))

    def apply(self, normal: ENNNormal) -> ENNNormal:
        from .enn_normal import ENNNormal

        mu = np.asarray(normal.mu, dtype=float)
        se_epi = np.asarray(normal.se_epi, dtype=float)
        se_ale = np.asarray(normal.se_ale, dtype=float)

        if mu.ndim == 1:
            raise ValueError(f"ENNNormal.mu must be 2D, got {mu.shape}")
        m = mu.shape[-1]
        if self.a.shape != (m,) or self.b.shape != (m,) or self.c.shape != (m,):
            raise ValueError(
                f"calibrator metrics {(self.a.shape, self.b.shape, self.c.shape)} "
                f"!= mu last axis {m}"
            )

        from enn._rust import affine_apply

        mu_p, se_p, se_epi_p, se_ale_p = affine_apply(
            self.a, self.b, self.c, mu, se_epi, se_ale, _bounds(normal.y_bounds)
        )
        return ENNNormal(
            mu=np.asarray(mu_p, dtype=float),
            se=np.asarray(se_p, dtype=float),
            se_epi=np.asarray(se_epi_p, dtype=float),
            se_ale=np.asarray(se_ale_p, dtype=float),
            idx=normal.idx,
            y_bounds=normal.y_bounds,
        )

    def map_mu(
        self,
        mu: np.ndarray,
        y_bounds: np.ndarray | None = None,
        se: np.ndarray | None = None,
    ) -> np.ndarray:
        mu_arr = np.asarray(mu, dtype=float)
        m = int(self.a.shape[0])
        if mu_arr.shape[-1] != m:
            raise ValueError(
                f"mu last axis {mu_arr.shape[-1]} != calibrator metrics {m}"
            )
        from enn._rust import affine_map_mu

        se_flat = None if se is None else np.asarray(_align(np.asarray(se), mu_arr.shape)).reshape(-1, m)
        out = affine_map_mu(
            self.a, self.b, mu_arr.reshape(-1, m), _bounds(y_bounds), se_flat
        )
        return np.asarray(out, dtype=float).reshape(mu_arr.shape)

    def map_draws(
        self,
        draws: np.ndarray,
        mu: np.ndarray,
        y_bounds: np.ndarray | None = None,
        se: np.ndarray | None = None,
    ) -> np.ndarray:
        draws_arr = np.asarray(draws, dtype=float)
        mu_arr = np.asarray(mu, dtype=float)
        m = int(self.a.shape[0])
        if draws_arr.shape[-1] != m:
            raise ValueError(
                f"draws last axis {draws_arr.shape[-1]} != calibrator metrics {m}"
            )
        if mu_arr.shape[-1] != m:
            raise ValueError(
                f"mu last axis {mu_arr.shape[-1]} != calibrator metrics {m}"
            )
        from enn._rust import affine_map_draws

        mu_b = _align(mu_arr, draws_arr.shape)
        se_flat = None if se is None else np.asarray(_align(np.asarray(se), draws_arr.shape)).reshape(-1, m)
        out = affine_map_draws(
            self.a,
            self.b,
            self.c,
            draws_arr.reshape(-1, m),
            np.asarray(mu_b).reshape(-1, m),
            _bounds(y_bounds),
            se_flat,
        )
        return np.asarray(out, dtype=float).reshape(draws_arr.shape)


def fit_affine_calibrator(
    model: Any,
    params: Any,
    *,
    num_samples: int,
    rng: Any,
    observation_noise: bool = False,
) -> AffineCalibrator:
    from enn._rust import fit_model_affine

    samples = int(num_samples)
    if samples < 0:
        samples = 0
    seed = int(rng.integers(0, 2**63 - 1))
    a, b, c = fit_model_affine(
        model.rust_backend,
        int(params.k_num_neighbors),
        float(params.epistemic_variance_scale),
        float(params.aleatoric_variance_scale),
        samples,
        seed,
        bool(observation_noise),
    )
    return AffineCalibrator(
        a=np.asarray(a, dtype=float),
        b=np.asarray(b, dtype=float),
        c=np.asarray(c, dtype=float),
    )
