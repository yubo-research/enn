from __future__ import annotations

from enum import Enum, auto


class ENNIndexDriver(Enum):
    FLAT = auto()
    BPANN_DISK = auto()


# Canonical strings for Rust (model and optimizer both accept lowercase)
ENN_INDEX_DRIVER_TO_RUST: dict[ENNIndexDriver, str] = {
    ENNIndexDriver.FLAT: "exact",
    ENNIndexDriver.BPANN_DISK: "bpann_disk",
}
