from __future__ import annotations

import os
from dataclasses import dataclass

from .enn_fit_config import ENNFitConfig
from .enn_index_driver import ENNIndexDriver


@dataclass(frozen=True)
class ENNSurrogateConfig:
    k: int | None = None
    fit: ENNFitConfig = ENNFitConfig()
    scale_x: bool = False
    index_driver: ENNIndexDriver = ENNIndexDriver.FLAT
    enn_storage: str | None = None
    work_dir: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        if self.scale_x and self.index_driver == ENNIndexDriver.BPANN_DISK:
            raise ValueError("scale_x=True is not compatible with BPANN_DISK")

    @property
    def num_fit_samples(self) -> int | None:
        return self.fit.num_fit_samples

    @property
    def num_fit_candidates(self) -> int | None:
        return self.fit.num_fit_candidates
