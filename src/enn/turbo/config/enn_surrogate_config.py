from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .enn_fit_config import ENNFitConfig
from .enn_index_driver import ENNIndexDriver
from .enn_x_scaling import ENNMetricLearning, ENNScaleX, validate_metric_learning, validate_scale_x


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
        validate_scale_x(self.scale_x, self.index_driver)
        validate_metric_learning(self.metric_learning, self.index_driver, self.scale_x)
        if self.y_bounds is not None:
            yb = np.asarray(self.y_bounds, dtype=float)
            if yb.ndim != 2 or yb.shape[1] != 2:
                raise ValueError(
                    f"y_bounds must have shape (num_metrics, 2), got {yb.shape}"
                )
            object.__setattr__(self, "y_bounds", yb)

    @property
    def num_fit_samples(self) -> int | None:
        return self.fit.num_fit_samples

    @property
    def num_fit_candidates(self) -> int | None:
        return self.fit.num_fit_candidates
