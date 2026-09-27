from __future__ import annotations

from enum import Enum, auto

from .enn_index_driver import ENNIndexDriver


class ENNXScaling(Enum):
    """How ENN distances weight the input dimensions.

    - ``NONE``: raw ``x``.
    - ``SCALE_X``: divide ``x`` by per-dimension data scales (FLAT only).
    - ``METRIC_LEARNING``: caller-set diagonal metric via ``MBPANNMetric`` (BPANN_DISK only).
    """

    NONE = auto()
    SCALE_X = auto()
    METRIC_LEARNING = auto()


_REQUIRED_DRIVER = {
    ENNXScaling.SCALE_X: ENNIndexDriver.FLAT,
    ENNXScaling.METRIC_LEARNING: ENNIndexDriver.BPANN_DISK,
}


def validate_x_scaling(x_scaling: ENNXScaling, index_driver: ENNIndexDriver) -> None:
    if not isinstance(x_scaling, ENNXScaling):
        raise ValueError(f"x_scaling must be an ENNXScaling, got {x_scaling!r}")
    required = _REQUIRED_DRIVER.get(x_scaling)
    if required is not None and index_driver != required:
        raise ValueError(f"x_scaling={x_scaling.name} requires index_driver={required.name}")
