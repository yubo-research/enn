from __future__ import annotations

import operator
from collections.abc import Sequence
from enum import Enum, auto

from .enn_index_driver import ENNIndexDriver


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
    flat = [j for g in groups for j in g]
    if any(len(g) == 0 for g in groups):
        raise ValueError("tied_dims groups must be non-empty")
    if any(j < 0 or j >= num_dim for j in flat):
        raise ValueError(f"tied_dims entries must be in [0, {num_dim}), got {flat}")
    if len(set(flat)) != len(flat):
        raise ValueError(f"tied_dims groups must be disjoint, got {groups}")
    return groups


def validate_scale_x(scale_x: ENNScaleX, index_driver: ENNIndexDriver) -> None:
    if not isinstance(scale_x, ENNScaleX):
        raise ValueError(f"scale_x must be an ENNScaleX, got {scale_x!r}")
    if not isinstance(index_driver, ENNIndexDriver):
        raise ValueError(f"index_driver must be an ENNIndexDriver, got {index_driver!r}")


def validate_metric_learning(
    metric_learning: ENNMetricLearning,
    index_driver: ENNIndexDriver,
    scale_x: ENNScaleX = ENNScaleX.OFF,
) -> None:
    if not isinstance(metric_learning, ENNMetricLearning):
        raise ValueError(f"metric_learning must be an ENNMetricLearning, got {metric_learning!r}")
    if metric_learning == ENNMetricLearning.NONE:
        return
    if index_driver != ENNIndexDriver.BPANN_DISK:
        raise ValueError(
            f"metric_learning={metric_learning.name} requires index_driver=BPANN_DISK, "
            f"got {index_driver.name}"
        )
    if scale_x == ENNScaleX.ON:
        raise ValueError(f"metric_learning={metric_learning.name} requires scale_x=OFF")
