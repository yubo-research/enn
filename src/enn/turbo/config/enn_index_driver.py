from __future__ import annotations

from enum import Enum, auto


class ENNIndexDriver(Enum):
    FLAT = auto()
    BPANN_DISK = auto()


def index_driver_to_wire(driver: ENNIndexDriver) -> str:
    if not isinstance(driver, ENNIndexDriver):
        raise ValueError(f"index_driver must be an ENNIndexDriver, got {driver!r}")
    return driver.name


def index_driver_from_wire(name: str) -> ENNIndexDriver:
    try:
        return ENNIndexDriver[name]
    except KeyError:
        raise ValueError(f"Unknown index_driver: {name}") from None
