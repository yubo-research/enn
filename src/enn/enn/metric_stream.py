"""Diagonal ENN metric weights from a row stream, without an optimizer.

The ENN distance is ``sum_d a_d (x_d - x'_d)^2``. The pieces used by ``ENNMetricLearning.AUTO``:

- ``Reservoir``: a uniform random sample of fixed size of every row seen so far
  (Algorithm R), so the rows used for metric fitting represent the whole stream.
- ``dependence_weights``: first-order Sobol index of ``y`` on each input (estimated by binning),
  divided by the input's variance. No optimization. A group of tied inputs (e.g. the one-hot
  columns of one categorical variable) shares one weight from its joint Sobol index, unscaled.
- ``auto_weights``: those weights and their leave-one-out gain over the best isotropic metric.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPS = 1e-9
MIN_VAR = 1e-24
DEPENDENCE_FLOOR = 1e-6
DEPENDENCE_Z = 3.0
MIN_DEPENDENCE_ROWS = 100
SCALE_GRID = tuple(10.0 ** np.arange(-2.0, 4.01, 0.5))
NOISE_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _as_rows(x: np.ndarray, y: np.ndarray, num_dim: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1, num_dim)
    y = np.asarray(y, dtype=float)
    y = y.reshape(-1, 1) if y.ndim < 2 else y
    if len(x) != len(y):
        raise ValueError(f"x has {len(x)} rows but y has {len(y)}")
    return x, y


def _spread(x: np.ndarray) -> np.ndarray:
    var = x.var(axis=0) if len(x) else np.ones(x.shape[1])
    return np.where(var > MIN_VAR, var, 1.0)


class Reservoir:
    """Uniform sample of up to ``capacity`` of the rows passed to ``add`` (Algorithm R).

    ``y`` rows have ``num_outputs`` columns; a 1-D ``y`` passed to ``add`` is one column.
    """

    def __init__(
        self, capacity: int, num_dim: int, rng: np.random.Generator, num_outputs: int = 1
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._x = np.empty((capacity, num_dim))
        self._y = np.empty((capacity, num_outputs))
        self._rng = rng
        self.num_seen = 0

    @property
    def capacity(self) -> int:
        return len(self._y)

    def __len__(self) -> int:
        return min(self.num_seen, self.capacity)

    @property
    def x(self) -> np.ndarray:
        return self._x[: len(self)]

    @property
    def y(self) -> np.ndarray:
        return self._y[: len(self)]

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        """Row ``t`` (0-based over the stream) replaces a random slot with probability ``capacity / (t + 1)``."""
        x, y = _as_rows(x, y, self._x.shape[1])
        if y.shape[1] != self._y.shape[1]:
            raise ValueError(f"y has {y.shape[1]} columns, expected {self._y.shape[1]}")
        start, num_fill = self.num_seen, max(0, min(len(y), self.capacity - self.num_seen))
        self._x[start : start + num_fill] = x[:num_fill]
        self._y[start : start + num_fill] = y[:num_fill]
        if len(y) > num_fill:
            slots = self._rng.integers(0, np.arange(start + num_fill, start + len(y)) + 1)
            for i in np.flatnonzero(slots < self.capacity):
                self._x[slots[i]] = x[num_fill + i]
                self._y[slots[i]] = y[num_fill + i]
        self.num_seen += len(y)


def sobol_index(x: np.ndarray, y: np.ndarray, num_bins: int | None = None) -> np.ndarray:
    """First-order Sobol index ``Var(E[y | x_d]) / Var(y)`` of each input, by binning (1-D ``y``).

    Each input is cut into ``num_bins`` equal-count bins (default ``floor(sqrt(n))``). The
    between-bin share of the sum of squares is corrected for the part expected from noise
    alone (the ANOVA "epsilon squared"), then floored at 0.
    """
    n, d = x.shape
    b = max(2, int(np.sqrt(n))) if num_bins is None else num_bins
    yc = y - y.mean()
    ss_tot = float(yc @ yc)
    if n < 2 * b or ss_tot <= 0:
        return np.zeros(d)
    ranks = np.argsort(np.argsort(x, axis=0, kind="stable"), axis=0, kind="stable")
    cells = (ranks * b // n + b * np.arange(d)).ravel()
    count = np.bincount(cells, minlength=b * d).reshape(d, b)
    total = np.bincount(cells, weights=np.repeat(yc[:, None], d, axis=1).ravel(), minlength=b * d).reshape(d, b)
    return _between_share(count, total, ss_tot, n, b)


def _between_share(count: np.ndarray, total: np.ndarray, ss_tot: float, n: int, b: int) -> np.ndarray:
    ss_between = (total**2 / np.maximum(count, 1)).sum(axis=-1)
    ms_within = (ss_tot - ss_between) / (n - b)
    return np.maximum(ss_between - (b - 1) * ms_within, 0.0) / ss_tot


def _cells(x_group: np.ndarray) -> tuple[np.ndarray, int]:
    _, cells = np.unique(x_group, axis=0, return_inverse=True)
    cells = cells.ravel()
    return cells, int(cells.max()) + 1 if len(cells) else 0


def group_sobol_index(x_group: np.ndarray, y: np.ndarray) -> float:
    """Joint first-order Sobol index ``Var(E[y | x_G]) / Var(y)`` of a group of discrete inputs (1-D ``y``).

    Each distinct row of ``x_group`` (for one-hot columns, each category) is one cell, and the
    between-cell share is corrected for noise as in ``sobol_index``. The index is 0 with one
    cell or fewer than two rows per cell.
    """
    n = len(y)
    cells, b = _cells(x_group)
    yc = y - y.mean()
    ss_tot = float(yc @ yc)
    if b < 2 or n < 2 * b or ss_tot <= 0:
        return 0.0
    count = np.bincount(cells, minlength=b)
    total = np.bincount(cells, weights=yc, minlength=b)
    return float(_between_share(count, total, ss_tot, n, b))


def null_sd(n: int, num_cells: int | None = None) -> float:
    """Approximate standard deviation of the Sobol index of an input ``y`` does not depend on (``n`` rows).

    ``num_cells`` is the number of bins or categories (default ``floor(sqrt(n))``, as in ``sobol_index``).
    """
    if n < 3:
        return np.inf
    b = max(2, int(np.sqrt(n))) if num_cells is None else num_cells
    return float(np.sqrt(2.0 * (b - 1)) / n)


def _passing(s: np.ndarray, n: int, num_cells: int | None = None) -> np.ndarray:
    return np.where(s > DEPENDENCE_Z * null_sd(n, num_cells), s, 0.0)


def _unit_indices(
    x: np.ndarray, y: np.ndarray, tied: Sequence[Sequence[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Passing Sobol index of each unit (an untied input or a tied group) and the unit of each input.

    Untied inputs come first, in order, then the groups.
    """
    n, d = x.shape
    unit = np.arange(d)
    s = _passing(np.mean([sobol_index(x, col) for col in y.T], axis=0), n)
    if not tied:
        return s, unit
    free = np.setdiff1d(unit, np.concatenate([list(g) for g in tied]))
    unit[free] = np.arange(len(free))
    joint = []
    for i, g in enumerate(tied):
        unit[list(g)] = len(free) + i
        s_g = np.mean([group_sobol_index(x[:, list(g)], col) for col in y.T])
        joint.append(_passing(s_g, n, _cells(x[:, list(g)])[1]))
    return np.concatenate([s[free], joint]), unit


def dependence_weights(
    x: np.ndarray,
    y: np.ndarray,
    floor: float = DEPENDENCE_FLOOR,
    tied: Sequence[Sequence[int]] = (),
) -> np.ndarray:
    """Weights ``a_d = U * S_u / (sum S * Var(x_d))`` from the Sobol index ``S`` of each of ``U`` units.

    A unit is an input, or a group in ``tied`` (disjoint lists of input indices). A group's
    ``S_u`` is its ``group_sobol_index``, and its inputs share one weight, not divided by
    ``Var(x_d)``. With several ``y`` columns, ``S`` is the mean of their indices. ``S_u`` below
    ``DEPENDENCE_Z`` null standard deviations counts as 0, then every ``S_u`` is raised to at
    least ``floor * max(S)``. If no unit passes, all ``S_u`` are equal and the weights are
    ``1 / Var(x_d)`` (the ``SCALE_X`` distances), and 1 for tied inputs.
    """
    x, y = _as_rows(x, y, np.shape(x)[-1])
    s, unit = _unit_indices(x, y, tied)
    s = np.maximum(s, floor * s.max()) if s.max() > 0 else np.ones_like(s)
    spread = _spread(x)
    for g in tied:
        spread[list(g)] = 1.0
    return (len(s) * s / s.sum())[unit] / spread


def _sq_dists(xq: np.ndarray, xr: np.ndarray, a: np.ndarray) -> np.ndarray:
    return (xq * xq) @ a[:, None] + (xr * xr) @ a - 2.0 * (xq * a) @ xr.T


def _enn_loglik(d2: np.ndarray, y_nbr: np.ndarray, yq: np.ndarray, c: float) -> np.ndarray:
    w = 1.0 / (EPS + np.maximum(d2, 0.0) + c)
    s = w.sum(axis=1)
    mu = (w * y_nbr).sum(axis=1) / s
    var = 1.0 / s + c
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (yq - mu) ** 2 / var


def loo_loglik(x: np.ndarray, y: np.ndarray, a: np.ndarray, k: int) -> float:
    """Leave-one-out mean ENN log-likelihood of standardized ``y`` under weights ``a``.

    Each row is predicted from its ``k`` nearest other rows. The common scale of ``a`` and the
    noise variance ``c`` are not fitted but taken as the best of ``SCALE_GRID`` x ``NOISE_GRID``,
    per ``y`` column; the result is the mean over columns. Neighbors do not depend on either.
    """
    x, y = _as_rows(x, y, np.shape(x)[-1])
    spread = _spread(x)
    z, yz = (x - x.mean(axis=0)) / np.sqrt(spread), (y - y.mean(axis=0)) / np.sqrt(_spread(y))
    d2 = _sq_dists(z, z, a * spread)
    np.fill_diagonal(d2, np.inf)
    kk = min(k, len(y) - 1)
    nbr = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
    d2n = np.take_along_axis(d2, nbr, axis=1)
    return float(
        np.mean(
            [
                max(float(_enn_loglik(d2n * f, col[nbr], col, c).mean()) for f in SCALE_GRID for c in NOISE_GRID)
                for col in yz.T
            ]
        )
    )


def auto_weights(
    x: np.ndarray, y: np.ndarray, k: int, tied: Sequence[Sequence[int]] = ()
) -> tuple[np.ndarray, float]:
    """``(dependence_weights, gain)``: ``gain`` is their ``loo_loglik`` minus that of the identity metric.

    ``loo_loglik`` picks the common scale, so the identity metric stands for the best isotropic
    metric. With fewer than ``MIN_DEPENDENCE_ROWS`` rows the comparison is too noisy (a
    one-input metric can win on 10 rows by chance), so the gain is ``-inf``. ``tied`` is passed
    to ``dependence_weights``.
    """
    x, y = _as_rows(x, y, np.shape(x)[-1])
    if len(y) < MIN_DEPENDENCE_ROWS:
        return np.ones(x.shape[1]), float("-inf")
    w = dependence_weights(x, y, tied=tied)
    return w, loo_loglik(x, y, w, k) - loo_loglik(x, y, np.ones(x.shape[1]), k)
