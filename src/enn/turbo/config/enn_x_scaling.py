from __future__ import annotations

from enum import Enum, auto

from .enn_index_driver import ENNIndexDriver


class ENNScaleX(Enum):
    """Whether ENN divides ``x`` by per-dimension data scales (running standard deviations).

    FLAT rebuilds its index under the current scales after every ``add``. BPANN_DISK updates
    them incrementally: it rescales the stored index in place when some scale moved by more
    than 1%, and re-partitions only when a scale drifted by more than a factor of 2 from the
    one the partition was built under.
    """

    OFF = auto()
    ON = auto()


class ENNMetricLearning(Enum):
    """Whether ENN distances use a learned diagonal metric (BPANN_DISK only).

    - ``NONE``: raw ``x``.
    - ``AUTO``: the model keeps a reservoir sample of the added rows and, as the data grow,
      refits weights ``Sobol index / Var(x)`` per input. They are applied only when
      leave-one-out validation shows they beat the best isotropic metric; otherwise distances
      are those of ``NONE`` (``MBPANNMetric``).
    """

    NONE = auto()
    AUTO = auto()


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
