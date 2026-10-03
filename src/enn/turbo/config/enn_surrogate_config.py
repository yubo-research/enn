from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .enn_fit_config import ENNFitConfig
from .enn_index_driver import ENNIndexDriver, index_driver_to_wire
from .enn_x_scaling import (
    ENNMetricLearning,
    ENNScaleX,
    metric_learning_to_wire,
    scale_x_to_wire,
)


class ENNStorage(Enum):
    MEMORY = auto()
    DISK = auto()


def enn_storage_to_wire(storage: ENNStorage) -> str:
    """Encode a storage kind. The wire name is the enum member name."""
    if not isinstance(storage, ENNStorage):
        raise ValueError(f"enn_storage must be an ENNStorage, got {storage!r}")
    return storage.name


def validate_enn_placement(
    *,
    index_driver: ENNIndexDriver,
    enn_storage: ENNStorage | None,
    work_dir: str | os.PathLike[str] | None,
    scale_x: ENNScaleX,
    metric_learning: ENNMetricLearning,
) -> None:
    """Check the placement options. Each wire encoder rejects a wrong type."""
    from enn._rust import validate_enn_placement as rust_validate

    rust_validate(
        index_driver_to_wire(index_driver),
        None if enn_storage is None else enn_storage_to_wire(enn_storage),
        None if work_dir is None else os.fspath(work_dir),
        scale_x_to_wire(scale_x),
        metric_learning_to_wire(metric_learning),
    )


@dataclass(frozen=True)
class ENNSurrogateConfig:
    k: int | None = None
    fit: ENNFitConfig = ENNFitConfig()
    scale_x: ENNScaleX = ENNScaleX.OFF
    metric_learning: ENNMetricLearning = ENNMetricLearning.NONE
    tied_dims: tuple = ()
    index_driver: ENNIndexDriver = ENNIndexDriver.FLAT
    enn_storage: ENNStorage | None = None
    work_dir: str | os.PathLike[str] | None = None
    y_bounds: np.ndarray | None = None

    def __post_init__(self) -> None:
        validate_enn_placement(
            index_driver=self.index_driver,
            enn_storage=self.enn_storage,
            work_dir=self.work_dir,
            scale_x=self.scale_x,
            metric_learning=self.metric_learning,
        )
        if self.y_bounds is not None:
            object.__setattr__(
                self, "y_bounds", np.asarray(self.y_bounds, dtype=float)
            )

    @property
    def num_fit_samples(self) -> int | None:
        return self.fit.num_fit_samples

    @property
    def num_fit_candidates(self) -> int | None:
        return self.fit.num_fit_candidates
