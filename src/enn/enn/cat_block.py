"""Rademacher blocks, one-hot blocks, and the shared hash."""

from __future__ import annotations

import numpy as np

_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_VAR = np.uint64(0x9E3779B97F4A7C15)
CODE_BIT = 64


def _mix(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(30))
    x = x * _MIX_A
    x = x ^ (x >> np.uint64(27))
    x = x * _MIX_B
    return x ^ (x >> np.uint64(31))


def _var_key(var: int, seed: int) -> np.uint64:
    wrapped = (int(var) * int(_VAR)) & ((1 << CODE_BIT) - 1)
    return np.uint64(seed) ^ np.uint64(wrapped)


def _ids(categories: np.ndarray, var: int, seed: int) -> np.ndarray:
    cats = np.ascontiguousarray(categories, dtype=np.int64).view(np.uint64)
    return _mix(cats ^ _var_key(var, seed))


def rademacher(
    categories: np.ndarray, var: int, width: int, seed: int, separation: float
) -> np.ndarray:
    hashed = _ids(categories, var, seed)
    shifts = np.arange(width, dtype=np.uint64)
    bits = ((hashed[:, None] >> shifts) & np.uint64(1)).astype(np.float64)
    return (bits * 2.0 - 1.0) * (float(separation) / np.sqrt(width))


def hash_columns(categories: np.ndarray, hash_dim: int, seed: int) -> np.ndarray:
    n_rows, n_vars = categories.shape
    out = np.zeros((n_rows, hash_dim), dtype=np.float64)
    if n_rows == 0 or n_vars == 0:
        return out
    rows = np.arange(n_rows)
    for var in range(n_vars):
        hashed = _ids(categories[:, var], var, seed)
        idx = (hashed % np.uint64(hash_dim)).astype(np.int64)
        sign = np.where(hashed >> np.uint64(63), 1.0, -1.0)
        np.add.at(out, (rows, idx), sign)
    return out / np.sqrt(n_vars)


def width_in_range(value: int, label: str, upper: int) -> int:
    width = int(value)
    if width < 1 or width > upper:
        raise ValueError(f"{label} must be in 1..{upper}, got {width}")
    return width


def log_code_width(card: int, dim_cap: int = 48) -> int:
    """Rademacher width for a domain of ``card`` categories.

    Before the cap, the width satisfies ``2**width >= 128 * card**2``, about
    one pair of categories in 200 sharing a vector.
    """
    count = int(card)
    cap = width_in_range(dim_cap, "dim_cap", CODE_BIT)
    if count < 1:
        raise ValueError(f"card must be >= 1, got {count}")
    if count == 1:
        return 1
    width = int(np.ceil(2.0 * np.log2(count) + 7.0))
    if width > cap:
        return cap
    return width


def write_onehot(out: np.ndarray, ids: np.ndarray, cursor: int, card: int) -> None:
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= card):
        raise ValueError(f"one-hot ids must be in 0..{card - 1}")
    block = out[:, cursor : cursor + card]
    block[:] = 0.0
    if ids.size:
        block[np.arange(ids.size), ids] = 1.0


def snap_onehot(block: np.ndarray, domain, card: int) -> np.ndarray:
    labels = np.asarray(domain, dtype=np.int64).ravel()
    if labels.size == 0:
        raise ValueError("domain must be non-empty")
    if int(labels.min()) < 0 or int(labels.max()) >= card:
        raise ValueError(f"one-hot domain ids must be in 0..{card - 1}")
    return labels[block[:, labels].argmax(axis=1)]


def snap_code(block, domain, var, width, seed, separation) -> np.ndarray:
    labels = np.asarray(domain, dtype=np.int64).ravel()
    if labels.size == 0:
        raise ValueError("domain must be non-empty")
    table = rademacher(labels, var, width, seed, separation)
    pick = (block @ table.T).argmax(axis=1)
    return labels[pick]
