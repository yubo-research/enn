from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TurboConfig:
    k: int | None = None
    num_candidates: int | None = None
    num_init: int | None = None
    var_scale: float = 1.0

    # Experimental
    trailing_obs: int | None = None
    num_fit_samples: int | None = None
    num_fit_candidates: int | None = None
    acq_type: Literal["thompson", "pareto", "ucb"] = "pareto"
    local_only: bool = False

    def __post_init__(self) -> None:
        if self.acq_type not in ["thompson", "pareto", "ucb"]:
            raise ValueError(
                f"acq_type must be 'thompson', 'pareto', or 'ucb', got {self.acq_type!r}"
            )
        # Pareto acquisition is the only type that works well without hyperparameter fitting
        if self.num_fit_samples is None and self.acq_type != "pareto":
            raise ValueError(f"num_fit_samples required for acq_type={self.acq_type!r}")
