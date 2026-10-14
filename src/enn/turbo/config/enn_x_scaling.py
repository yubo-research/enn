from __future__ import annotations

import operator
from collections.abc import Sequence
from enum import Enum, auto

from enn._rust import validate_tied_dims as _validate_tied_dims


class ENNScaleX(Enum):

    OFF = auto()
    ON = auto()


class ENNMetricLearning(Enum):

    NONE = auto()
    AUTO = auto()


def scale_x_to_wire(scale_x: ENNScaleX) -> bool:
    if not isinstance(scale_x, ENNScaleX):
        raise TypeError(f"scale_x must be an ENNScaleX, got {scale_x!r}")
    return scale_x is ENNScaleX.ON


def metric_learning_to_wire(metric_learning: ENNMetricLearning) -> str:
    if not isinstance(metric_learning, ENNMetricLearning):
        raise TypeError(
            f"metric_learning must be an ENNMetricLearning, got {metric_learning!r}"
        )
    return metric_learning.name


def validate_tied_dims(tied_dims: Sequence[Sequence[int]] | None, num_dim: int) -> tuple[tuple[int, ...], ...]:
    if tied_dims is None:
        return ()
    groups = tuple(tuple(operator.index(j) for j in g) for g in tied_dims)
    _validate_tied_dims([list(g) for g in groups], int(num_dim))
    return groups
