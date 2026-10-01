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


def validate_enn_placement(
    *,
    index_driver: ENNIndexDriver,
    enn_storage: ENNStorage | None,
    work_dir: str | os.PathLike[str] | None,
    scale_x: ENNScaleX,
    metric_learning: ENNMetricLearning,
) -> None:
    from enn._rust import validate_enn_placement as rust_validate

    if not isinstance(index_driver, ENNIndexDriver):
        raise ValueError(f"index_driver must be an ENNIndexDriver, got {index_driver!r}")
    if enn_storage is not None and not isinstance(enn_storage, ENNStorage):
        raise ValueError(f"enn_storage must be an ENNStorage, got {enn_storage!r}")
    if not isinstance(scale_x, ENNScaleX):
        raise ValueError(f"scale_x must be an ENNScaleX, got {scale_x!r}")
    if not isinstance(metric_learning, ENNMetricLearning):
        raise ValueError(
            f"metric_learning must be an ENNMetricLearning, got {metric_learning!r}"
        )
    rust_validate(
        index_driver.name,
        None if enn_storage is None else enn_storage.name,
        None if work_dir is None else os.fspath(work_dir),
        scale_x is ENNScaleX.ON,
        "auto" if metric_learning is ENNMetricLearning.AUTO else "none",
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
