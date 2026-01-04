from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrustRegionConfig:
    tr_type: Literal["turbo", "morbo", "none"] = "turbo"
    num_metrics: int | None = None

    def __post_init__(self) -> None:
        if self.tr_type not in {"turbo", "morbo", "none"}:
            raise ValueError(
                f"tr_type must be 'turbo', 'morbo', or 'none', got {self.tr_type!r}"
            )
        if self.num_metrics is not None and self.num_metrics < 1:
            raise ValueError(f"num_metrics must be >= 1, got {self.num_metrics}")
        if self.tr_type in {"turbo", "none"} and self.num_metrics not in (None, 1):
            raise ValueError(
                f"num_metrics must be 1 for tr_type={self.tr_type!r}, got {self.num_metrics}"
            )


@dataclass(frozen=True)
class CandidateGenConfig:
    candidate_rv: Literal["sobol", "uniform"] = "sobol"
    num_candidates: int | None = None

    def __post_init__(self) -> None:
        if self.candidate_rv not in {"sobol", "uniform"}:
            raise ValueError(
                f"candidate_rv must be 'sobol' or 'uniform', got {self.candidate_rv!r}"
            )
        if self.num_candidates is not None and self.num_candidates <= 0:
            raise ValueError(f"num_candidates must be > 0, got {self.num_candidates}")


@dataclass(frozen=True)
class InitConfig:
    init_strategy: Literal["hybrid", "lhd_only"] = "hybrid"
    num_init: int | None = None

    def __post_init__(self) -> None:
        if self.init_strategy not in {"hybrid", "lhd_only"}:
            raise ValueError(
                f"init_strategy must be 'hybrid' or 'lhd_only', got {self.init_strategy!r}"
            )
        if self.num_init is not None and self.num_init <= 0:
            raise ValueError(f"num_init must be > 0, got {self.num_init}")
