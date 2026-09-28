from __future__ import annotations

from enum import Enum, auto

from .enn_index_driver import ENNIndexDriver


class ENNScaleX(Enum):
    """Whether ENN divides ``x`` by per-dimension data scales (FLAT only)."""

    OFF = auto()
    ON = auto()


class ENNMetricLearning(Enum):
    """Whether ENN distances use a learned diagonal metric (BPANN_DISK only).

    - ``OFF``: raw ``x``.
    - ``ON``: caller-set diagonal metric via ``MBPANNMetric``.
    - ``AUTO``: like ``ON``, but a fitted metric is applied only when held-out validation shows
      it beats the best isotropic metric; otherwise distances are those of ``OFF``
      (``MBPANNMetric.set_weights_if_validated``).
    """

    OFF = auto()
    ON = auto()
    AUTO = auto()


def validate_scale_x(scale_x: ENNScaleX, index_driver: ENNIndexDriver) -> None:
    if not isinstance(scale_x, ENNScaleX):
        raise ValueError(f"scale_x must be an ENNScaleX, got {scale_x!r}")
    if scale_x == ENNScaleX.ON and index_driver != ENNIndexDriver.FLAT:
        raise ValueError(f"scale_x=ON requires index_driver=FLAT, got {index_driver.name}")


def validate_metric_learning(
    metric_learning: ENNMetricLearning, index_driver: ENNIndexDriver
) -> None:
    if not isinstance(metric_learning, ENNMetricLearning):
        raise ValueError(f"metric_learning must be an ENNMetricLearning, got {metric_learning!r}")
    if metric_learning != ENNMetricLearning.OFF and index_driver != ENNIndexDriver.BPANN_DISK:
        raise ValueError(
            f"metric_learning={metric_learning.name} requires index_driver=BPANN_DISK, "
            f"got {index_driver.name}"
        )
