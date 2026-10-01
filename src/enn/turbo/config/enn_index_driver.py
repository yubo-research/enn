from __future__ import annotations

from enum import Enum, auto


class ENNIndexDriver(Enum):
    FLAT = auto()
    BPANN_DISK = auto()



ENN_INDEX_DRIVER_TO_RUST: dict[ENNIndexDriver, str] = {
    ENNIndexDriver.FLAT: "FLAT",
    ENNIndexDriver.BPANN_DISK: "BPANN_DISK",
}
