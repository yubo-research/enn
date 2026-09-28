"""Diagonal ENN metric weights from a row stream, without a batch optimizer.

The ENN distance is ``sum_d a_d (x_d - x'_d)^2``. Three pieces, all fed row by row:

- ``Reservoir``: a uniform random sample of fixed size of every row seen so far
  (Algorithm R), so the rows used for metric fitting represent the whole stream.
- ``dependence_weights``: weights from a measure of how much ``y`` depends on each input
  (squared correlation, or a first-order Sobol index estimated by binning), divided by the
  input's variance. No optimization; ``validated_dependence_weights`` keeps them only if they
  beat ``1 / Var(x)`` in leave-one-out log-likelihood on the same rows.
- ``PerturbationMetricLearner``: keeps ``theta = (log a, log c)`` in standardized units and
  moves it a little on every new row by simultaneous perturbation (SPSA): the row is predicted
  from its ``k`` nearest reservoir rows under ``theta + p Delta`` and ``theta - p Delta``, and
  the difference of the two log-likelihoods estimates the gradient. The row is scored before
  it can enter the reservoir, so every score is out of sample.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9
MIN_VAR = 1e-24
LOG_BOUND = 12.0
DEPENDENCE_FLOOR = 1e-6
DEPENDENCE_Z = 3.0
DEPENDENCE_INDICES = ("correlation", "sobol")
CANDIDATES_PER_NEIGHBOR = 4
MIN_DEPENDENCE_ROWS = 100
SCALE_GRID = tuple(10.0 ** np.arange(-2.0, 4.01, 0.5))
NOISE_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _as_rows(x: np.ndarray, y: np.ndarray, num_dim: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1, num_dim)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError(f"x has {len(x)} rows but y has {len(y)}")
    return x, y


def _spread(x: np.ndarray) -> np.ndarray:
    var = x.var(axis=0) if len(x) else np.ones(x.shape[1])
    return np.where(var > MIN_VAR, var, 1.0)


class Reservoir:
    """Uniform sample of up to ``capacity`` of the rows passed to ``add`` (Algorithm R)."""

    def __init__(self, capacity: int, num_dim: int, rng: np.random.Generator) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._x = np.empty((capacity, num_dim))
        self._y = np.empty(capacity)
        self._rng = rng
        self.num_seen = 0
        self.version = 0

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
        """Row ``t`` (0-based over the stream) replaces a random slot with probability ``capacity / (t + 1)``.

        ``version`` increases whenever a row is stored, so callers can cache statistics of the sample.
        """
        x, y = _as_rows(x, y, self._x.shape[1])
        start, num_fill = self.num_seen, max(0, min(len(y), self.capacity - self.num_seen))
        self._x[start : start + num_fill] = x[:num_fill]
        self._y[start : start + num_fill] = y[:num_fill]
        stored = num_fill
        if len(y) > num_fill:
            slots = self._rng.integers(0, np.arange(start + num_fill, start + len(y)) + 1)
            for i in np.flatnonzero(slots < self.capacity):
                self._x[slots[i]] = x[num_fill + i]
                self._y[slots[i]] = y[num_fill + i]
                stored += 1
        self.version += stored > 0
        self.num_seen += len(y)


def correlation_index(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared Pearson correlation of ``y`` with each input, less its null bias ``(1 - r^2) / (n - 2)``, floored at 0."""
    n = len(y)
    if n < 3:
        return np.zeros(x.shape[1])
    xc, yc = x - x.mean(axis=0), y - y.mean()
    denom = np.sqrt((xc * xc).sum(axis=0) * (yc @ yc))
    r2 = np.where(denom > 0, (xc.T @ yc) ** 2 / np.where(denom > 0, denom, 1.0) ** 2, 0.0)
    return np.maximum(r2 - (1.0 - r2) / (n - 2), 0.0)


def sobol_index(x: np.ndarray, y: np.ndarray, num_bins: int | None = None) -> np.ndarray:
    """First-order Sobol index ``Var(E[y | x_d]) / Var(y)`` of each input, by binning.

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
    ss_between = (total**2 / np.maximum(count, 1)).sum(axis=1)
    ms_within = (ss_tot - ss_between) / (n - b)
    return np.maximum(ss_between - (b - 1) * ms_within, 0.0) / ss_tot


def null_sd(index: str, n: int) -> float:
    """Approximate standard deviation of the index of an input ``y`` does not depend on (``n`` rows)."""
    if n < 3:
        return np.inf
    dof = 1 if index == "correlation" else max(2, int(np.sqrt(n))) - 1
    return float(np.sqrt(2.0 * dof) / n)


def dependence_weights(
    x: np.ndarray, y: np.ndarray, index: str = "sobol", floor: float = DEPENDENCE_FLOOR
) -> np.ndarray:
    """Weights ``a_d = D * S_d / (sum S * Var(x_d))`` from a dependence index ``S`` (``D`` inputs).

    ``S_d`` below ``DEPENDENCE_Z`` null standard deviations counts as 0, then every ``S_d`` is
    raised to at least ``floor * max(S)``. If no input passes, all ``S_d`` are equal and the
    weights are ``1 / Var(x_d)`` (the ``SCALE_X`` distances).
    """
    if index not in DEPENDENCE_INDICES:
        raise ValueError(f"index must be one of {DEPENDENCE_INDICES}, got {index!r}")
    x, y = _as_rows(x, y, np.shape(x)[-1])
    s = correlation_index(x, y) if index == "correlation" else sobol_index(x, y)
    s = np.where(s > DEPENDENCE_Z * null_sd(index, len(y)), s, 0.0)
    s = np.maximum(s, floor * s.max()) if s.max() > 0 else np.ones_like(s)
    return len(s) * s / s.sum() / _spread(x)


def _sq_dists(xq: np.ndarray, xr: np.ndarray, a: np.ndarray, xr_sq: np.ndarray | None = None) -> np.ndarray:
    xr_sq = xr * xr if xr_sq is None else xr_sq
    return (xq * xq) @ a[:, None] + xr_sq @ a - 2.0 * (xq * a) @ xr.T


def _enn_loglik(d2: np.ndarray, y_nbr: np.ndarray, yq: np.ndarray, c: np.ndarray | float) -> np.ndarray:
    c_col = np.reshape(c, (-1, 1))
    w = 1.0 / (EPS + np.maximum(d2, 0.0) + c_col)
    s = w.sum(axis=1)
    mu = (w * y_nbr).sum(axis=1) / s
    var = 1.0 / s + c_col[:, 0]
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (yq - mu) ** 2 / var


def loo_loglik(x: np.ndarray, y: np.ndarray, a: np.ndarray, k: int) -> float:
    """Leave-one-out mean ENN log-likelihood of standardized ``y`` under weights ``a``.

    Each row is predicted from its ``k`` nearest other rows. The common scale of ``a`` and the
    noise variance ``c`` are not fitted but taken as the best of ``SCALE_GRID`` x ``NOISE_GRID``;
    neighbors do not depend on either.
    """
    spread, y_spread = _spread(x), float(_spread(y[:, None])[0])
    z, yz = (x - x.mean(axis=0)) / np.sqrt(spread), (y - y.mean()) / np.sqrt(y_spread)
    d2 = _sq_dists(z, z, a * spread)
    np.fill_diagonal(d2, np.inf)
    kk = min(k, len(y) - 1)
    nbr = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
    d2n, yn = np.take_along_axis(d2, nbr, axis=1), yz[nbr]
    return max(float(_enn_loglik(d2n * f, yn, yz, c).mean()) for f in SCALE_GRID for c in NOISE_GRID)


def validated_dependence_weights(x: np.ndarray, y: np.ndarray, k: int, index: str = "sobol") -> np.ndarray:
    """``dependence_weights`` if their ``loo_loglik`` is higher than that of ``1 / Var(x)``, else ``1 / Var(x)``.

    With fewer than ``MIN_DEPENDENCE_ROWS`` rows the check itself is too noisy (a one-input
    metric can win on 10 rows by chance), so ``1 / Var(x)`` is returned.
    """
    x, y = _as_rows(x, y, np.shape(x)[-1])
    flat = 1.0 / _spread(x)
    if len(y) < MIN_DEPENDENCE_ROWS:
        return flat
    dep = dependence_weights(x, y, index)
    return dep if loo_loglik(x, y, dep, k) > loo_loglik(x, y, flat, k) else flat


def perturbed_loglik(
    xq: np.ndarray,
    yq: np.ndarray,
    xr: np.ndarray,
    yr: np.ndarray,
    thetas: list[np.ndarray],
    k: int,
    xr_sq: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Gaussian log-likelihood of each query row predicted by ENN from its ``k`` nearest rows of ``xr``.

    ``thetas[j][i] = (log a, log c)`` for query ``i`` under setting ``j``. Candidates are the
    ``CANDIDATES_PER_NEIGHBOR * k`` rows nearest under the mean of the settings; each
    setting chooses its ``k`` neighbors among them. ``xr_sq`` optionally supplies ``xr * xr``.
    """
    center = np.mean([t.mean(axis=0) for t in thetas], axis=0)
    num_cand = min(len(yr), CANDIDATES_PER_NEIGHBOR * k)
    d2 = _sq_dists(xq, xr, np.exp(center[:-1]), xr_sq).astype(np.float32)
    cand = np.argpartition(d2, num_cand - 1, axis=1)[:, :num_cand]
    diff2, ycand = (xq[:, None, :] - xr[cand]) ** 2, yr[cand]
    kk = min(k, num_cand)
    rows = np.arange(len(yq))[:, None]
    out = []
    for t in thetas:
        dc = np.einsum("mcd,md->mc", diff2, np.exp(t[:, :-1]))
        nbr = np.argpartition(dc, kk - 1, axis=1)[:, :kk]
        out.append(_enn_loglik(dc[rows, nbr], ycand[rows, nbr], yq, np.exp(t[:, -1])))
    return out


class PerturbationMetricLearner:
    """Stateful SPSA fit of the ENN metric; ``update`` moves it a little for every new row.

    State: ``theta = (log a_1 .. log a_d, log c)`` in standardized units (inputs and targets
    divided by their reservoir standard deviations), running second moment ``v`` of the
    gradient estimates, and the reservoir. For the ``t``-th new row, with a random sign vector
    ``Delta``, ``g = (ll(theta + p Delta) - ll(theta - p Delta)) / (2 p) * Delta`` and
    ``theta += step / sqrt(1 + t / decay_rows) * g / sqrt(v)``. Rows passed in one call share the
    state at the start of the call (the sum of their steps is applied at once); with one row per
    call the update is exactly per row. ``theta`` starts at 0 (equal weights in standardized
    units, the ``SCALE_X`` distances); with ``init_rows > 0`` it is reset once to
    ``validated_dependence_weights`` of the reservoir when the reservoir first holds ``init_rows`` rows.
    """

    def __init__(
        self,
        num_dim: int,
        rng: np.random.Generator,
        *,
        k: int = 10,
        capacity: int = 1000,
        perturbation: float = 0.6,
        step: float = 0.05,
        decay_rows: float = 1000.0,
        init_rows: int = 0,
    ) -> None:
        if not (perturbation > 0 and step > 0 and decay_rows > 0 and k >= 1 and init_rows >= 0):
            raise ValueError("need perturbation > 0, step > 0, decay_rows > 0, k >= 1, init_rows >= 0")
        self.reservoir = Reservoir(capacity, num_dim, rng)
        self.theta = np.concatenate([np.zeros(num_dim), [np.log(0.1)]])
        self._v: np.ndarray | None = None
        self._rng = rng
        self._k, self._p, self._step, self._decay_rows = k, perturbation, step, decay_rows
        self._init_rows = init_rows
        self._cache_version = -1
        self._cache: tuple = ()
        self.num_updates = 0

    @property
    def weights(self) -> np.ndarray:
        """Metric weights in the raw input units."""
        return np.exp(self.theta[:-1]) / _spread(self.reservoir.x)

    def _maybe_init(self) -> None:
        if self._init_rows and len(self.reservoir) >= self._init_rows:
            xr = self.reservoir.x
            a = validated_dependence_weights(xr, self.reservoir.y, self._k) * _spread(xr)
            self.theta[:-1] = np.clip(np.log(a), -LOG_BOUND, LOG_BOUND)
            self._init_rows = 0

    def _standardized_reservoir(self) -> tuple:
        """``(mean, std, zr, zr * zr, yz, y_mean, y_std)`` of the reservoir, recomputed only when it has changed."""
        if self._cache_version != self.reservoir.version:
            xr, yr = self.reservoir.x, self.reservoir.y
            mean, std = xr.mean(axis=0), np.sqrt(_spread(xr))
            y_mean, y_std = yr.mean(), float(np.sqrt(_spread(yr[:, None])[0]))
            zr = (xr - mean) / std
            self._cache = (mean, std, zr, zr * zr, (yr - y_mean) / y_std, y_mean, y_std)
            self._cache_version = self.reservoir.version
        return self._cache

    def _gradient_estimates(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mean, std, zr, zr_sq, yz, y_mean, y_std = self._standardized_reservoir()
        delta = self._rng.choice((-1.0, 1.0), size=(len(y), len(self.theta)))
        ll_plus, ll_minus = perturbed_loglik(
            (x - mean) / std,
            (y - y_mean) / y_std,
            zr,
            yz,
            [self.theta + self._p * delta, self.theta - self._p * delta],
            self._k,
            zr_sq,
        )
        return ((ll_plus - ll_minus) / (2.0 * self._p))[:, None] * delta

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Score the new rows against the reservoir, step ``theta``, then offer the rows to the reservoir."""
        x, y = _as_rows(x, y, len(self.theta) - 1)
        self._maybe_init()
        if len(self.reservoir) >= 2 and len(y):
            g = self._gradient_estimates(x, y)
            g2 = (g * g).mean(axis=0)
            decay = 0.999 ** len(y)
            self._v = g2 if self._v is None else decay * self._v + (1 - decay) * g2
            rate = self._step / np.sqrt(1.0 + (self.num_updates + np.arange(len(y))) / self._decay_rows)
            step = (rate[:, None] * g).sum(axis=0) / (np.sqrt(self._v) + EPS)
            self.theta = np.clip(self.theta + step, -LOG_BOUND, LOG_BOUND)
            self.num_updates += len(y)
        self.reservoir.add(x, y)
