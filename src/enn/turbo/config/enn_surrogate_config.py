from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .enn_fit_config import ENNFitConfig
from .enn_index_driver import ENNIndexDriver
from .enn_x_scaling import ENNMetricLearning, ENNScaleX


@dataclass(frozen=True)
class ENNSurrogateConfig:
    k: int | None = None
    fit: ENNFitConfig = ENNFitConfig()
    scale_x: ENNScaleX = ENNScaleX.OFF
    metric_learning: ENNMetricLearning = ENNMetricLearning.NONE
    tied_dims: tuple = ()
    index_driver: ENNIndexDriver = ENNIndexDriver.FLAT
    enn_storage: str | None = None
    work_dir: str | os.PathLike[str] | None = None
    y_bounds: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scale_x, ENNScaleX):
            raise ValueError(f"scale_x must be an ENNScaleX, got {self.scale_x!r}")
        if not isinstance(self.index_driver, ENNIndexDriver):
            raise ValueError(
                f"index_driver must be an ENNIndexDriver, got {self.index_driver!r}"
            )
        if not isinstance(self.metric_learning, ENNMetricLearning):
            raise ValueError(
                f"metric_learning must be an ENNMetricLearning, got {self.metric_learning!r}"
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
