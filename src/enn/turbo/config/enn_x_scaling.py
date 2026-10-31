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


def validate_tied_dims(
    tied_dims: Sequence[Sequence[int]] | None, num_dim: int
) -> tuple[tuple[int, ...], ...]:
    if tied_dims is None:
        return ()
    groups = tuple(tuple(operator.index(j) for j in g) for g in tied_dims)
    _validate_tied_dims([list(g) for g in groups], int(num_dim))
    return groups
