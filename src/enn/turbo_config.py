from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class TurboConfig:
    k: Optional[int] = None
    num_candidates: Optional[int] = None
    num_init: Optional[int] = None
    var_scale: float = 1.0

    # Experimental
    trailing_obs: Optional[int] = None
    num_fit_samples: Optional[int] = None
    acq_type: Literal["thompson", "pareto", "ucb"] = "pareto"
