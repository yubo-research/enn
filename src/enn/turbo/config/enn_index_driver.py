from __future__ import annotations

from enum import Enum, auto


class ENNIndexDriver(Enum):
    FLAT = auto()
    FAST_MEM = auto()
    BPANN_DISK = auto()



ENN_INDEX_DRIVER_TO_RUST: dict[ENNIndexDriver, str] = {
    ENNIndexDriver.FLAT: "exact",
    ENNIndexDriver.FAST_MEM: "fast_mem",
    ENNIndexDriver.BPANN_DISK: "bpann_disk",
}
