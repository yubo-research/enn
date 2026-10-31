from __future__ import annotations

import numpy as np


class _EncodedRows(np.ndarray):
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


def resolve_tied_groups(values, tied_dims, *, semantic_only=False):
    from enn._rust import resolve_tied_groups as resolve

    groups = resolve(
        [list(g) for g in _groups_on(values)],
        None if tied_dims is None else [list(g) for g in tied_dims],
        semantic_only,
    )
    return tuple(tuple(g) for g in groups)


def attach_tied_dims(
    values: np.ndarray, tied_dims: tuple[tuple[int, ...], ...]
) -> np.ndarray:
    return _EncodedRows(values, tied_dims)
