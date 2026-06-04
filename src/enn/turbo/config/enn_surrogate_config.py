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

    @property
    def num_fit_samples(self) -> int | None:
        return self.fit.num_fit_samples

    @property
    def num_fit_candidates(self) -> int | None:
        return self.fit.num_fit_candidates
