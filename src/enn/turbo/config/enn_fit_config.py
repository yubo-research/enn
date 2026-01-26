from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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
