from __future__ import annotations

from enum import Enum, auto


class ENNIndexDriver(Enum):
    FLAT = auto()
    BPANN_DISK = auto()
    MBPANN_DISK = auto()



ENN_INDEX_DRIVER_TO_RUST: dict[ENNIndexDriver, str] = {
    ENNIndexDriver.FLAT: "exact",
    ENNIndexDriver.BPANN_DISK: "bpann_disk",
    ENNIndexDriver.MBPANN_DISK: "mbpann_disk",
}

DISK_INDEX_DRIVERS = (ENNIndexDriver.BPANN_DISK, ENNIndexDriver.MBPANN_DISK)
