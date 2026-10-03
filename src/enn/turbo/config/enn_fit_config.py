from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ENNFitConfig:
    """Scale-search settings. ``num_fit_samples=None`` freezes the scales, and then the
    other settings do not apply. ``infer_aleatoric_variance_scale=None`` means True."""

    num_fit_samples: int | None = None
    num_fit_candidates: int | None = None
    infer_aleatoric_variance_scale: bool | None = None
    affine_calibrate: bool = False

    def __post_init__(self) -> None:
        if self.num_fit_samples is not None and self.num_fit_samples <= 0:
            raise ValueError(f"num_fit_samples must be > 0, got {self.num_fit_samples}")
        if self.num_fit_candidates is not None and self.num_fit_candidates <= 0:
            raise ValueError(
                f"num_fit_candidates must be > 0, got {self.num_fit_candidates}"
            )
        if self.num_fit_samples is None:
            search_only = {
                "num_fit_candidates": self.num_fit_candidates is not None,
                "infer_aleatoric_variance_scale": self.infer_aleatoric_variance_scale
                is not None,
                "affine_calibrate": self.affine_calibrate,
            }
            set_names = [name for name, is_set in search_only.items() if is_set]
            if set_names:
                raise ValueError(
                    f"{', '.join(set_names)} require num_fit_samples; "
                    "num_fit_samples=None freezes the scales"
                )

    @property
    def infers_aleatoric_variance(self) -> bool:
        return self.infer_aleatoric_variance_scale is not False
