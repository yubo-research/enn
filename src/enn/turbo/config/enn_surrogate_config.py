from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from .enn_index_driver import ENNIndexDriver
from .enn_fit_config import ENNFitConfig

if TYPE_CHECKING:
    from ..components.protocols import Surrogate


@dataclass(frozen=True)
class ENNSurrogateConfig:
    k: int | None = None
    fit: ENNFitConfig = ENNFitConfig()
    scale_x: bool = False
    index_driver: ENNIndexDriver = ENNIndexDriver.FLAT

    @property
    def num_fit_samples(self) -> int | None:
        return self.fit.num_fit_samples

    @property
    def num_fit_candidates(self) -> int | None:
        return self.fit.num_fit_candidates

    def build(self) -> Surrogate:
        from ..components.builder import build_surrogate

        return build_surrogate(self)
