from __future__ import annotations

import numpy as np


class AppendableArray:
    def __init__(self, initial_capacity: int = 100) -> None:
        self._initial_capacity = initial_capacity
        self._buffer: np.ndarray | None = None
        self._size = 0
        self._num_cols: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        if self._num_cols is None:
            return (0, 0)
        return (self._size, self._num_cols)

    def _initialize_buffer(self, row: np.ndarray) -> None:
        if row.ndim == 0:
            self._num_cols = 1
            row = row.reshape(1, 1)
        elif row.ndim == 1:
            self._num_cols = row.shape[0]
            row = row[np.newaxis, :]
        elif row.ndim == 2:
            if row.shape[0] != 1:
                raise ValueError(f"Expected row shape (1, D), got {row.shape}")
            self._num_cols = row.shape[1]
        else:
            raise ValueError(f"Expected 0D, 1D or 2D array, got {row.ndim}D")

        self._buffer = np.empty(
            (self._initial_capacity, self._num_cols), dtype=row.dtype
        )

    def _validate_row(self, row: np.ndarray) -> np.ndarray:
        if row.ndim == 0:
            if self._num_cols != 1:
                raise ValueError(f"Expected {self._num_cols} columns, got 1 (scalar)")
            return row.reshape(1, 1)
        if row.ndim == 1:
            if row.shape[0] != self._num_cols:
                raise ValueError(
                    f"Expected {self._num_cols} columns, got {row.shape[0]}"
                )
            return row[np.newaxis, :]
        if row.ndim == 2:
            if row.shape != (1, self._num_cols):
                raise ValueError(
                    f"Expected shape (1, {self._num_cols}), got {row.shape}"
                )
            return row
        raise ValueError(f"Expected 0D, 1D or 2D array, got {row.ndim}D")

    def append(self, row: np.ndarray) -> None:
        row = np.asarray(row)

        if self._num_cols is None:
            self._initialize_buffer(row)

        row = self._validate_row(row)

        assert self._buffer is not None
        if self._size + 1 > self._buffer.shape[0]:
            new_capacity = self._buffer.shape[0] * 2
            new_buffer = np.empty(
                (new_capacity, self._num_cols), dtype=self._buffer.dtype
            )
            new_buffer[: self._size] = self._buffer[: self._size]
            self._buffer = new_buffer

        self._buffer[self._size] = row
        self._size += 1

    def view(self) -> np.ndarray:
        if self._buffer is None:
            return np.empty((0, 0))
        return self._buffer[: self._size]

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, key) -> np.ndarray:
        return self.view()[key]
