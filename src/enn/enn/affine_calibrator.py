from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .enn_normal import ENNNormal

_BOUND_EPS = 1e-3
_BOUND_SE_MULT = 0.25


def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {out.shape}")
    return out


def _ols_ab(mu: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, m = mu.shape
    a = np.zeros(m, dtype=float)
    b = np.ones(m, dtype=float)
    if n < 2:
        return a, b
    for j in range(m):
        mj = mu[:, j]
        yj = y[:, j]
        s_mu = float(mj.sum())
        s_y = float(yj.sum())
        s_mumu = float(np.dot(mj, mj))
        s_muy = float(np.dot(mj, yj))
        det = n * s_mumu - s_mu * s_mu
        if not np.isfinite(det) or abs(det) < 1e-18 * max(1.0, abs(n * s_mumu)):
            continue
        a[j] = (s_mumu * s_y - s_mu * s_muy) / det
        b[j] = (n * s_muy - s_mu * s_y) / det
        if not (np.isfinite(a[j]) and np.isfinite(b[j])):
            a[j], b[j] = 0.0, 1.0
    return a, b


def _residual_c(
    mu: np.ndarray,
    se: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    m = mu.shape[1]
    c = np.ones(m, dtype=float)
    pred = a.reshape(1, -1) + b.reshape(1, -1) * mu
    resid = y - pred
    for j in range(m):
        rms_r = float(np.sqrt(np.mean(resid[:, j] ** 2)))
        rms_s = float(np.sqrt(np.mean(se[:, j] ** 2)))
        if not np.isfinite(rms_s) or rms_s <= 0.0 or not np.isfinite(rms_r):
            c[j] = 1.0
        else:
            c[j] = rms_r / rms_s
    return c


def _column_margin(se_col: np.ndarray | None, span: float) -> np.ndarray | float:
    floor = max(_BOUND_EPS, 1e-6 * span if np.isfinite(span) else _BOUND_EPS)
    if se_col is None:
        return floor
    return np.maximum(floor, _BOUND_SE_MULT * np.abs(se_col))


def _project_mu_to_open_bounds(
    mu: np.ndarray,
    y_bounds: np.ndarray | None,
    se: np.ndarray | None = None,
) -> np.ndarray:
    if y_bounds is None:
        return mu
    bounds = np.asarray(y_bounds, dtype=float)
    if bounds.size == 0:
        return mu
    if np.all(np.isneginf(bounds[:, 0]) & np.isposinf(bounds[:, 1])):
        return mu
    out = np.array(mu, dtype=float, copy=True)
    se_arr = None if se is None else np.asarray(se, dtype=float)
    for j in range(bounds.shape[0]):
        lo = float(bounds[j, 0])
        hi = float(bounds[j, 1])
        col = out[..., j]
        se_col = None if se_arr is None else se_arr[..., j]
        if np.isfinite(lo) and np.isfinite(hi):
            eps = _column_margin(se_col, hi - lo)
            lo_b = lo + eps
            hi_b = hi - eps
            mid = 0.5 * (lo + hi)
            out[..., j] = np.where(lo_b >= hi_b, mid, np.clip(col, lo_b, hi_b))
        elif np.isfinite(lo):
            out[..., j] = np.maximum(col, lo + _column_margin(se_col, np.inf))
        elif np.isfinite(hi):
            out[..., j] = np.minimum(col, hi - _column_margin(se_col, np.inf))
    return out


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
        a, b = _ols_ab(mu2, y2)
        if fit_residual_scale:
            if se is None:
                raise ValueError("se is required when fit_residual_scale=True")
            se2 = _as_2d(se, "se")
            if se2.shape != mu2.shape:
                raise ValueError(f"se shape {se2.shape} != mu shape {mu2.shape}")
            c = _residual_c(mu2, se2, y2, a, b)
        else:
            c = np.ones(mu2.shape[1], dtype=float)
        return cls(a=a, b=b, c=c)

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

        c = self.c.reshape((1,) * (mu.ndim - 1) + (m,))

        se_epi_p = c * se_epi
        se_ale_p = c * se_ale
        se_p = np.sqrt(se_epi_p * se_epi_p + se_ale_p * se_ale_p)
        mu_p = self.map_mu(mu, y_bounds=normal.y_bounds, se=se_p)

        return ENNNormal(
            mu=mu_p,
            se=se_p,
            se_epi=se_epi_p,
            se_ale=se_ale_p,
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
        shape = (1,) * (mu_arr.ndim - 1) + (m,)
        out = self.a.reshape(shape) + self.b.reshape(shape) * mu_arr
        return _project_mu_to_open_bounds(out, y_bounds, se=se)

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
        while mu_arr.ndim < draws_arr.ndim:
            mu_arr = np.expand_dims(mu_arr, axis=-2)
        shape = (1,) * (draws_arr.ndim - 1) + (m,)
        a = self.a.reshape(shape)
        b = self.b.reshape(shape)
        c = self.c.reshape(shape)
        out = a + b * mu_arr + c * (draws_arr - mu_arr)
        se_arr = se
        if se_arr is not None:
            se_arr = np.asarray(se_arr, dtype=float)
            while se_arr.ndim < draws_arr.ndim:
                se_arr = np.expand_dims(se_arr, axis=-2)
        return _project_mu_to_open_bounds(out, y_bounds, se=se_arr)


def fit_affine_calibrator(
    model: Any,
    params: Any,
    *,
    num_samples: int,
    rng: Any,
    observation_noise: bool = False,
) -> AffineCalibrator:
    from .enn_params import PosteriorFlags

    n = len(model)
    num_metrics = int(model.num_outputs)
    if n < 2 or num_samples <= 0:
        return AffineCalibrator.identity(num_metrics)

    p = min(int(num_samples), n)
    idx = np.asarray(rng.choice(n, size=p, replace=False), dtype=int)
    x_loo, y_loo, _yvar = model.train_rows_at(idx.tolist())
    flags = PosteriorFlags(exclude_nearest=True, observation_noise=observation_noise)
    post = model.posterior(x_loo, params=params, flags=flags)
    return AffineCalibrator.fit(post.mu, y_loo, post.se, fit_residual_scale=True)
