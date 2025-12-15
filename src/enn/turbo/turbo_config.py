from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TurboConfig:
    k: int | None = None
    num_candidates: int | None = None
    num_init: int | None = None
    num_fit_samples: int | None = None
    num_fit_candidates: int | None = None

    # Experimental
    trailing_obs: int | None = None
    tr_type: Literal["turbo", "morbo", "none"] = "turbo"
    eps_tr: float = 0.1
    acq_type: Literal["thompson", "pareto", "ucb"] = "pareto"
    num_metrics: int | None = None

    def __post_init__(self) -> None:
        if self.acq_type not in ["thompson", "pareto", "ucb"]:
            raise ValueError(
                f"acq_type must be 'thompson', 'pareto', or 'ucb', got {self.acq_type!r}"
            )
        if self.num_fit_samples is None and self.acq_type != "pareto":
            raise ValueError(f"num_fit_samples required for acq_type={self.acq_type!r}")
        if self.tr_type not in ["turbo", "morbo", "none"]:
            raise ValueError(
                f"tr_type must be 'turbo', 'morbo', or 'none', got {self.tr_type!r}"
            )
        if self.num_metrics is not None and self.num_metrics < 1:
            raise ValueError(f"num_metrics must be >= 1, got {self.num_metrics}")
        eps_tr = float(self.eps_tr)
        if eps_tr < 0.0 or eps_tr > 1.0:
            raise ValueError(f"eps_tr must be in [0, 1], got {eps_tr}")
