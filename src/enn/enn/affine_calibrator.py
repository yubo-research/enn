from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .enn_normal import ENNNormal


def _column(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        return out.reshape(-1, 1)
    return out


def _rows(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        return out.reshape(-1, 1), out.shape
    return out.reshape(-1, out.shape[-1]), out.shape


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
        mu2 = _column(mu)
        y2 = _column(y)
        from enn._rust import fit_affine

        se_arr = None if se is None or not fit_residual_scale else _column(se)
        a, b, c = fit_affine(mu2, y2, se_arr)
        return cls(
            a=np.asarray(a, dtype=float),
            b=np.asarray(b, dtype=float),
            c=np.asarray(c, dtype=float),
        )

    def apply(self, normal: ENNNormal) -> ENNNormal:
        from .enn_normal import ENNNormal

        mu = np.asarray(normal.mu, dtype=float)
        se_epi = np.asarray(normal.se_epi, dtype=float)
        se_ale = np.asarray(normal.se_ale, dtype=float)

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
        rows, shape = _rows(mu_arr)
        from enn._rust import affine_map_mu

        se_flat = None
        if se is not None:
            se_flat = _rows(np.asarray(_align(np.asarray(se), mu_arr.shape)))[0]
        out = affine_map_mu(self.a, self.b, rows, _bounds(y_bounds), se_flat)
        return np.asarray(out, dtype=float).reshape(shape)

    def map_draws(
        self,
        draws: np.ndarray,
        mu: np.ndarray,
        y_bounds: np.ndarray | None = None,
        se: np.ndarray | None = None,
    ) -> np.ndarray:
        draws_arr = np.asarray(draws, dtype=float)
        mu_arr = np.asarray(mu, dtype=float)
        from enn._rust import affine_map_draws

        draws_rows, shape = _rows(draws_arr)
        mu_rows = _rows(np.asarray(_align(mu_arr, draws_arr.shape)))[0]
        se_flat = None
        if se is not None:
            se_flat = _rows(np.asarray(_align(np.asarray(se), draws_arr.shape)))[0]
        out = affine_map_draws(
            self.a,
            self.b,
            self.c,
            draws_rows,
            mu_rows,
            _bounds(y_bounds),
            se_flat,
        )
        return np.asarray(out, dtype=float).reshape(shape)


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
