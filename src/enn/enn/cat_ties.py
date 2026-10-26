"""Tie groups carried on arrays from ``CatEncoder.encode``."""

from __future__ import annotations

import operator

import numpy as np


class _EncodedRows(np.ndarray):
    """Encoded rows. ``tied_dims`` groups the columns of one categorical variable."""

    tied_dims: tuple[tuple[int, ...], ...] = ()

    def __new__(cls, values: np.ndarray, tied_dims: tuple[tuple[int, ...], ...]):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.tied_dims = tied_dims
        return obj

    def __array_finalize__(self, obj) -> None:
        self.tied_dims = _groups_if_same_columns(self, obj)


def _groups_if_same_columns(view, source) -> tuple[tuple[int, ...], ...]:
    if not isinstance(source, _EncodedRows):
        return ()
    if view.ndim != 2 or source.ndim != 2:
        return ()
    same_width = view.shape[1] == source.shape[1]
    same_stride = view.strides[1] == source.strides[1]
    if same_width and same_stride:
        return source.tied_dims
    return ()


def _groups_on(values) -> tuple[tuple[int, ...], ...]:
    carried = getattr(values, "tied_dims", None)
    if isinstance(carried, tuple):
        return carried
    return ()


def resolve_tied_groups(values, tied_dims):
    """Grouping for metric learning.

    Empty ``tied_dims`` keeps the categorical blocks on ``values``. An
    explicit grouping must list those same blocks.
    """
    carried = _groups_on(values)
    if tied_dims is None or len(tied_dims) == 0:
        return carried
    explicit = tuple(tuple(operator.index(j) for j in group) for group in tied_dims)
    if carried and explicit != carried:
        raise ValueError("tied_dims do not match the categorical blocks on x")
    return explicit


def attach_tied_dims(
    values: np.ndarray, tied_dims: tuple[tuple[int, ...], ...]
) -> np.ndarray:
    """Return ``values`` labeled with ``tied_dims``."""
    return _EncodedRows(values, tied_dims)
