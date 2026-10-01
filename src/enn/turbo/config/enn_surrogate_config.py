from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .enn_fit_config import ENNFitConfig
from .enn_index_driver import ENNIndexDriver
from .enn_x_scaling import ENNMetricLearning, ENNScaleX


class ENNStorage(Enum):
    MEMORY = auto()
    DISK = auto()


def _reject_auto(
    index_driver: ENNIndexDriver,
    enn_storage: ENNStorage | None,
    scale_x: ENNScaleX,
    metric_learning: ENNMetricLearning,
) -> None:
    if metric_learning != ENNMetricLearning.AUTO:
        return
    legal = (
        index_driver is ENNIndexDriver.BPANN_DISK
        and enn_storage is ENNStorage.DISK
        and scale_x is ENNScaleX.OFF
    )
    if not legal:
        raise ValueError(
            "metric_learning=Auto requires IndexDriver::BpAnnDisk, "
            "disk storage, and scale_x=false"
        )


def validate_enn_placement(
    *,
    index_driver: ENNIndexDriver,
    enn_storage: ENNStorage | None,
    work_dir: str | os.PathLike[str] | None,
    scale_x: ENNScaleX,
    metric_learning: ENNMetricLearning,
) -> None:
    if not isinstance(index_driver, ENNIndexDriver):
        raise ValueError(f"index_driver must be an ENNIndexDriver, got {index_driver!r}")
    if enn_storage is not None and not isinstance(enn_storage, ENNStorage):
        raise ValueError(f"enn_storage must be an ENNStorage, got {enn_storage!r}")
    if work_dir is not None and enn_storage is not ENNStorage.DISK:
        raise ValueError("work_dir does not select storage; pass ENNStorage.DISK explicitly")
    if enn_storage is ENNStorage.DISK and index_driver is not ENNIndexDriver.BPANN_DISK:
        raise ValueError("Disk storage requires IndexDriver::BpAnnDisk")
    _reject_auto(index_driver, enn_storage, scale_x, metric_learning)


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
        if not isinstance(self.scale_x, ENNScaleX):
            raise ValueError(f"scale_x must be an ENNScaleX, got {self.scale_x!r}")
        if not isinstance(self.metric_learning, ENNMetricLearning):
            raise ValueError(
                f"metric_learning must be an ENNMetricLearning, got {self.metric_learning!r}"
            )
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
