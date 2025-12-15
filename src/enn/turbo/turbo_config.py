from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TurboConfig:
    k: int | None = None
    num_candidates: int | None = None
    num_init: int | None = None

    # Experimental
    trailing_obs: int | None = None
    tr_type: Literal["turbo", "morbo", "none"] = "turbo"
    num_metrics: int | None = None

    def __post_init__(self) -> None:
        if self.tr_type not in ["turbo", "morbo", "none"]:
            raise ValueError(
                f"tr_type must be 'turbo', 'morbo', or 'none', got {self.tr_type!r}"
            )
        if self.num_metrics is not None and self.num_metrics < 1:
            raise ValueError(f"num_metrics must be >= 1, got {self.num_metrics}")
        if self.tr_type == "turbo":
            if self.num_metrics is not None and self.num_metrics != 1:
                raise ValueError(
                    f"num_metrics must be 1 for tr_type='turbo', got {self.num_metrics}"
                )
        if self.tr_type == "none":
            if self.num_metrics is not None and self.num_metrics != 1:
                raise ValueError(
                    f"num_metrics must be 1 for tr_type='none', got {self.num_metrics}"
                )


@dataclass(frozen=True)
class TurboOneConfig(TurboConfig):
    pass


@dataclass(frozen=True)
class TurboZeroConfig(TurboConfig):
    pass


@dataclass(frozen=True)
class LHDOnlyConfig(TurboConfig):
    pass


@dataclass(frozen=True)
class TurboENNConfig(TurboConfig):
    acq_type: Literal["thompson", "pareto", "ucb"] = "pareto"
    num_fit_samples: int | None = None
    num_fit_candidates: int | None = None
    scale_x: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.acq_type not in ["thompson", "pareto", "ucb"]:
            raise ValueError(
                f"acq_type must be 'thompson', 'pareto', or 'ucb', got {self.acq_type!r}"
            )
        if self.num_fit_samples is None and self.acq_type != "pareto":
            raise ValueError(f"num_fit_samples required for acq_type={self.acq_type!r}")
        if self.num_fit_samples is not None and int(self.num_fit_samples) <= 0:
            raise ValueError(f"num_fit_samples must be > 0, got {self.num_fit_samples}")
        if self.num_fit_candidates is not None and int(self.num_fit_candidates) <= 0:
            raise ValueError(
                f"num_fit_candidates must be > 0, got {self.num_fit_candidates}"
            )
