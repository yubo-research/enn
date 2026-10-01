"""Input checks for optimizer tell()."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import TellInputs

if TYPE_CHECKING:
    import numpy as np


def validate_tell_inputs(
    x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None, num_dim: int
) -> TellInputs:
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != num_dim:
        raise ValueError(x.shape)
    if y.ndim == 2:
        if y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        num_metrics = y.shape[1]
    elif y.ndim == 1:
        if y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        num_metrics = 1
    else:
        raise ValueError(y.shape)
    if y_var is not None:
        y_var = np.asarray(y_var, dtype=float)
        if y_var.shape != y.shape:
            raise ValueError((y.shape, y_var.shape))
    return TellInputs(x=x, y=y, y_var=y_var, num_metrics=num_metrics)
