from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


def standardize_y(y: np.ndarray | list[float] | Any) -> tuple[float, float]:
    import numpy as np

    y_array = np.asarray(y, dtype=float)
    center = float(np.median(y_array))
    scale = float(np.std(y_array))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return center, scale


def _validate_sobol_inputs(x, y):
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    n, d = x.shape
    if d <= 0:
        raise ValueError(f"x must have at least 1 dimension, got {d}")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(f"y shape {y.shape} incompatible with x rows {n}")
    return n, d, y


def _compute_sobol_bins(x, y, n, d):
    import numpy as np

    B = 10 if n >= 30 else 3
    order = np.argsort(x, axis=0)
    row_idx = np.arange(n).reshape(n, 1).repeat(d, axis=1)
    ranks = np.empty_like(order)
    ranks[order, np.arange(d)[None, :]] = row_idx
    idx = (ranks * B) // n
    oh = np.zeros((n, d, B), dtype=x.dtype)
    oh[np.arange(n)[:, None], np.arange(d)[None, :], idx] = 1.0
    counts, sums = oh.sum(axis=0), (oh * y.reshape(n, 1, 1)).sum(axis=0)
    mu_b = np.zeros_like(sums)
    mask = counts > 0
    mu_b[mask] = sums[mask] / counts[mask]
    return counts / float(n), mu_b


def calculate_sobol_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    import numpy as np

    n, d, y = _validate_sobol_inputs(x, y)
    if n < 9:
        return np.ones(d, dtype=x.dtype)
    mu, vy = y.mean(), y.var(ddof=0)
    if not np.isfinite(vy) or vy <= 0:
        return np.ones(d, dtype=x.dtype)
    p_b, mu_b = _compute_sobol_bins(x, y, n, d)
    S = (p_b * (mu_b - mu) ** 2).sum(axis=1) / vy
    return np.where(x.var(axis=0, ddof=0) <= 1e-12, np.zeros_like(S), S)


def pareto_front_2d_maximize(
    a: np.ndarray | Any, b: np.ndarray | Any, idx: np.ndarray | Any | None = None
) -> np.ndarray:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError((a.shape, b.shape))
    if idx is None:
        idx = np.arange(a.size, dtype=int)
    else:
        idx = np.asarray(idx, dtype=int)
        if idx.ndim != 1:
            raise ValueError(idx.shape)
    order = np.lexsort((-b[idx], -a[idx]))
    sorted_idx = idx[order]
    keep: list[int] = []
    best_b = -float("inf")
    last_a = float("nan")
    last_b = float("nan")
    for i in sorted_idx.tolist():
        bi = float(b[i])
        ai = float(a[i])
        if bi > best_b:
            keep.append(i)
            best_b = bi
            last_a = ai
            last_b = bi
        elif bi == best_b and ai == last_a and bi == last_b:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def arms_from_pareto_fronts(
    x_cand: np.ndarray | Any,
    mu: np.ndarray | Any,
    se: np.ndarray | Any,
    num_arms: int,
    rng: Generator | Any,
) -> np.ndarray:
    import numpy as np

    if x_cand.ndim != 2:
        raise ValueError(x_cand.shape)
    if mu.shape != se.shape or mu.ndim != 1:
        raise ValueError((mu.shape, se.shape))
    if mu.size != x_cand.shape[0]:
        raise ValueError((mu.size, x_cand.shape[0]))
    num_arms = int(num_arms)
    if num_arms <= 0:
        raise ValueError(num_arms)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(se)):
        raise ValueError("mu and se must be finite")
    i_keep: list[int] = []
    remaining = np.arange(mu.size, dtype=int)
    while remaining.size > 0 and len(i_keep) < num_arms:
        front_indices = pareto_front_2d_maximize(mu, se, remaining)
        if front_indices.size == 0:
            raise RuntimeError("pareto front extraction failed")
        front_indices = front_indices[np.argsort(-mu[front_indices])]
        if len(i_keep) + int(front_indices.size) <= num_arms:
            i_keep.extend(front_indices.tolist())
            is_front = np.zeros(mu.size, dtype=bool)
            is_front[front_indices] = True
            remaining = remaining[~is_front[remaining]]
            continue
        remaining_arms = num_arms - len(i_keep)
        i_keep.extend(
            rng.choice(front_indices, size=remaining_arms, replace=False).tolist()
        )
        break
    i_keep = np.array(i_keep)
    return x_cand[i_keep[np.argsort(-mu[i_keep])]]
