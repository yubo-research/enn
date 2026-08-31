"""Post-hoc affine calibrator: mu' = a + b*mu with residual SE scale c."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .enn_normal import ENNNormal


def _as_2d(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        return out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {out.shape}")
    return out


def _ols_ab(mu: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-metric OLS y ≈ a + b*mu. Returns (a, b) each shape (m,)."""
    n, m = mu.shape
    a = np.zeros(m, dtype=float)
    b = np.ones(m, dtype=float)
    if n < 2:
        return a, b
    ones = np.ones(n, dtype=float)
    for j in range(m):
        phi = np.column_stack([ones, mu[:, j]])
        coef, _, rank, _ = np.linalg.lstsq(phi, y[:, j], rcond=None)
        if rank >= 2:
            a[j], b[j] = float(coef[0]), float(coef[1])
        elif rank == 1:
            a[j] = float(np.mean(y[:, j] - mu[:, j]))
            b[j] = 1.0
    return a, b


def _residual_c(mu: np.ndarray, se: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """c = rms(y - (a+b*mu)) / rms(se) per metric; identity when undefined."""
    m = mu.shape[1]
    c = np.ones(m, dtype=float)
    resid = y - (a.reshape(1, -1) + b.reshape(1, -1) * mu)
    for j in range(m):
        rms_r = float(np.sqrt(np.mean(resid[:, j] ** 2)))
        rms_s = float(np.sqrt(np.mean(se[:, j] ** 2)))
        if rms_s > 0.0 and np.isfinite(rms_r) and np.isfinite(rms_s):
            c[j] = rms_r / rms_s
    return c


@dataclass
class AffineCalibrator:
    """Thin affine map on ENNNormal: mu' = a + b*mu; se_*' = c * se_*."""

    a: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    b: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=float))
    c: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=float))
    _n: float = 0.0
    _se_n: float = 0.0
    _s_mu: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    _s_y: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    _s_mumu: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    _s_muy: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    _s_y2: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    _s_se2: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))

    @classmethod
    def identity(cls, num_metrics: int = 1) -> AffineCalibrator:
        m = max(int(num_metrics), 1)
        return cls(
            a=np.zeros(m, dtype=float),
            b=np.ones(m, dtype=float),
            c=np.ones(m, dtype=float),
            _s_mu=np.zeros(m, dtype=float),
            _s_y=np.zeros(m, dtype=float),
            _s_mumu=np.zeros(m, dtype=float),
            _s_muy=np.zeros(m, dtype=float),
            _s_y2=np.zeros(m, dtype=float),
            _s_se2=np.zeros(m, dtype=float),
        )

    def fit(self, mu: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu2 = _as_2d(mu)
        y2 = _as_2d(y)
        if mu2.shape != y2.shape:
            raise ValueError(f"mu shape {mu2.shape} != y shape {y2.shape}")
        self.a, self.b = _ols_ab(mu2, y2)
        self._reset_gram_from_batch(mu2, y2)
        return self.a.copy(), self.b.copy()

    def update(
        self, mu: np.ndarray, y: np.ndarray, se: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rank-1 (or batch) RLS update; same solution as batch OLS on the multiset.

        When ``se`` is provided for every accumulated row (``_se_n == _n``), residual
        scale ``c`` is refreshed from moments in O(m), not O(P).
        """
        mu2 = _as_2d(mu)
        y2 = _as_2d(y)
        if mu2.shape != y2.shape:
            raise ValueError(f"mu shape {mu2.shape} != y shape {y2.shape}")
        se2 = None
        if se is not None:
            se2 = _as_2d(se)
            if se2.shape != y2.shape:
                raise ValueError(f"se shape {se2.shape} != y shape {y2.shape}")
        m = mu2.shape[1]
        self._ensure_metric_width(m)
        for i in range(mu2.shape[0]):
            self._n += 1.0
            self._s_mu += mu2[i]
            self._s_y += y2[i]
            self._s_mumu += mu2[i] * mu2[i]
            self._s_muy += mu2[i] * y2[i]
            self._s_y2 += y2[i] * y2[i]
            if se2 is not None:
                self._se_n += 1.0
                self._s_se2 += se2[i] * se2[i]
        self._solve_from_gram()
        self._refresh_c_from_moments()
        return self.a.copy(), self.b.copy()

    def fit_residual_scale(
        self, mu: np.ndarray, se: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        mu2 = _as_2d(mu)
        se2 = _as_2d(se)
        y2 = _as_2d(y)
        if mu2.shape != y2.shape or se2.shape != y2.shape:
            raise ValueError("mu, se, and y must share the same shape")
        n, m = mu2.shape
        self._ensure_metric_width(m)
        self._se_n = float(n)
        self._s_se2 = (se2 * se2).sum(axis=0)
        if self._n != float(n):
            self._n = float(n)
            self._s_mu = mu2.sum(axis=0)
            self._s_y = y2.sum(axis=0)
            self._s_mumu = (mu2 * mu2).sum(axis=0)
            self._s_muy = (mu2 * y2).sum(axis=0)
            self._s_y2 = (y2 * y2).sum(axis=0)
            self._solve_from_gram()
        elif not np.allclose(self._s_y2, (y2 * y2).sum(axis=0)):
            self._s_y2 = (y2 * y2).sum(axis=0)
        self._refresh_c_from_moments()
        if self._se_n != self._n or self._n < 1.0:
            self.c = _residual_c(mu2, se2, y2, self.a, self.b)
        return self.c.copy()

    def apply(self, normal: ENNNormal) -> ENNNormal:
        mu = np.asarray(normal.mu, dtype=float)
        se_epi = np.asarray(normal.se_epi, dtype=float)
        se_ale = np.asarray(normal.se_ale, dtype=float)
        a, b, c = self._broadcast_abc(mu.shape)
        mu_c = a + b * mu
        se_epi_c = c * se_epi
        se_ale_c = c * se_ale
        se_c = np.hypot(se_epi_c, se_ale_c)
        return ENNNormal(
            mu_c,
            se_c,
            se_epi_c,
            se_ale_c,
            idx=normal.idx,
            y_bounds=normal.y_bounds,
        )

    def _broadcast_abc(self, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = shape[-1]
        self._ensure_metric_width(m)
        pad = (1,) * (len(shape) - 1) + (m,)
        return self.a.reshape(pad), self.b.reshape(pad), self.c.reshape(pad)

    def _ensure_metric_width(self, m: int) -> None:
        if self.a.shape[0] == m:
            return
        if self._n == 0.0 and self.a.shape[0] == 1 and m > 1:
            self.a = np.zeros(m, dtype=float)
            self.b = np.ones(m, dtype=float)
            self.c = np.ones(m, dtype=float)
            self._s_mu = np.zeros(m, dtype=float)
            self._s_y = np.zeros(m, dtype=float)
            self._s_mumu = np.zeros(m, dtype=float)
            self._s_muy = np.zeros(m, dtype=float)
            self._s_y2 = np.zeros(m, dtype=float)
            self._s_se2 = np.zeros(m, dtype=float)
            self._se_n = 0.0
            return
        raise ValueError(f"calibrator width {self.a.shape[0]} != num_metrics {m}")

    def _reset_gram_from_batch(self, mu: np.ndarray, y: np.ndarray) -> None:
        n, m = mu.shape
        self._n = float(n)
        self._se_n = 0.0
        self._s_mu = mu.sum(axis=0)
        self._s_y = y.sum(axis=0)
        self._s_mumu = (mu * mu).sum(axis=0)
        self._s_muy = (mu * y).sum(axis=0)
        self._s_y2 = (y * y).sum(axis=0)
        self._s_se2 = np.zeros(m, dtype=float)
        self.c = np.ones(m, dtype=float)

    def _solve_from_gram(self) -> None:
        m = self._s_mu.shape[0]
        a = np.zeros(m, dtype=float)
        b = np.ones(m, dtype=float)
        if self._n < 2.0:
            self.a, self.b = a, b
            return
        for j in range(m):
            s = np.array(
                [[self._n, self._s_mu[j]], [self._s_mu[j], self._s_mumu[j]]],
                dtype=float,
            )
            r = np.array([self._s_y[j], self._s_muy[j]], dtype=float)
            try:
                coef = np.linalg.solve(s, r)
                a[j], b[j] = float(coef[0]), float(coef[1])
            except np.linalg.LinAlgError:
                a[j] = float(self._s_y[j] / self._n - self._s_mu[j] / self._n)
                b[j] = 1.0
        self.a, self.b = a, b

    def _refresh_c_from_moments(self) -> None:
        """c from Σ(y-a-bμ)² / Σse² without scanning stored pairs."""
        m = self._s_mu.shape[0]
        if self._n < 1.0 or self._se_n != self._n:
            return
        c = np.ones(m, dtype=float)
        n = self._n
        a, b = self.a, self.b
        sum_r2 = (
            self._s_y2
            + n * a * a
            + b * b * self._s_mumu
            - 2.0 * a * self._s_y
            - 2.0 * b * self._s_muy
            + 2.0 * a * b * self._s_mu
        )
        sum_r2 = np.maximum(sum_r2, 0.0)
        for j in range(m):
            rms_r = float(np.sqrt(sum_r2[j] / n))
            rms_s = float(np.sqrt(self._s_se2[j] / n))
            if rms_s > 0.0 and np.isfinite(rms_r) and np.isfinite(rms_s):
                c[j] = rms_r / rms_s
        self.c = c
