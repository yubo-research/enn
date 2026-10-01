from __future__ import annotations

import operator
from collections.abc import Sequence
from enum import Enum, auto

from enn._rust import validate_tied_dims as _validate_tied_dims


class ENNScaleX(Enum):
    """Whether ENN divides ``x`` by per-dimension data scales (running standard deviations).

    FLAT rebuilds its index under the current scales after every ``add``. BPANN_DISK updates
    them incrementally: it rescales the stored index in place when some scale moved by more
    than 1%, and re-partitions only when a scale drifted by more than a factor of 2 from the
    one the partition was built under. Dimensions in ``tied_dims`` keep scale 1.
    """

    OFF = auto()
    ON = auto()


class ENNMetricLearning(Enum):
    """Whether ENN distances use a learned diagonal metric (BPANN_DISK only).

    - ``NONE``: raw ``x``.
    - ``AUTO``: the model keeps a reservoir sample of the added rows and, as the data grow,
      refits weights ``Sobol index / Var(x)`` per input. They are applied only when
      leave-one-out validation shows they beat the best isotropic metric; otherwise distances
      are those of ``NONE`` (``MBPANNMetric``). Each group in ``tied_dims`` gets one weight
      from its joint Sobol index, not divided by ``Var(x)``, shared by its dimensions.
    """

    NONE = auto()
    AUTO = auto()


def validate_tied_dims(tied_dims: Sequence[Sequence[int]] | None, num_dim: int) -> tuple[tuple[int, ...], ...]:
    """Groups of tied input dimensions (e.g. the one-hot columns of one categorical variable).

    ``scale_x`` leaves tied dimensions unscaled, and ``ENNMetricLearning.AUTO`` gives each group
    one weight from its joint Sobol index. Groups must be non-empty and disjoint.
    """
    if tied_dims is None:
        return ()
    groups = tuple(tuple(operator.index(j) for j in g) for g in tied_dims)
    _validate_tied_dims([list(g) for g in groups], int(num_dim))
    return groups
