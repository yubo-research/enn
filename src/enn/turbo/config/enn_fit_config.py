from __future__ import annotations

from dataclasses import dataclass

from enn._rust import validate_fit_settings


@dataclass(frozen=True)
class ENNFitConfig:
    num_fit_samples: int | None = None
    num_fit_candidates: int | None = None
    infer_aleatoric_variance_scale: bool | None = None
    affine_calibrate: bool = False

    def __post_init__(self) -> None:
        validate_fit_settings(
            self.num_fit_samples,
            self.num_fit_candidates,
            self.infer_aleatoric_variance_scale,
            self.affine_calibrate,
        )

    @property
    def infers_aleatoric_variance(self) -> bool:
        return self.infer_aleatoric_variance_scale is not False
