from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from .enums import ENNIndexDriver

if TYPE_CHECKING:
    from ..components.protocols import Surrogate


@dataclass(frozen=True)
class ENNFitConfig:
    num_fit_samples: int | None = None
    num_fit_candidates: int | None = None

    def __post_init__(self) -> None:
        if self.num_fit_samples is not None and self.num_fit_samples <= 0:
            raise ValueError(f"num_fit_samples must be > 0, got {self.num_fit_samples}")
        if self.num_fit_candidates is not None and self.num_fit_candidates <= 0:
            raise ValueError(
                f"num_fit_candidates must be > 0, got {self.num_fit_candidates}"
            )


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
        from ..components.surrogates import ENNSurrogate

        return ENNSurrogate(self)
