from __future__ import annotations

from enum import Enum, auto


class ENNIndexDriver(Enum):
    FLAT = auto()
    BPANN_DISK = auto()


def index_driver_to_wire(driver: ENNIndexDriver) -> str:
    """Encode an index driver. The wire name is the enum member name."""
    if not isinstance(driver, ENNIndexDriver):
        raise ValueError(f"index_driver must be an ENNIndexDriver, got {driver!r}")
    return driver.name


def index_driver_from_wire(name: str) -> ENNIndexDriver:
    """Decode an index-driver wire name."""
    try:
        return ENNIndexDriver[name]
    except KeyError:
        raise ValueError(f"Unknown index_driver: {name}") from None
