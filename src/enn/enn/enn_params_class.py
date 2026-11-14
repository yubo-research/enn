from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from enn._rust import ENNParams as _validated_params

if TYPE_CHECKING:
    from .affine_calibrator import AffineCalibrator


def calibrator_coefficients(calibrator):
    if calibrator is None:
        return None
    return (calibrator.a, calibrator.b, calibrator.c)


@dataclass(frozen=True)
class ENNParams:
    k_num_neighbors: int
    epistemic_variance_scale: float
    aleatoric_variance_scale: float
    calibrator: AffineCalibrator | None = field(default=None, hash=False, compare=False)
    tau: float = 0.0

    def __post_init__(self) -> None:
        _validated_params(
            int(self.k_num_neighbors),
            float(self.epistemic_variance_scale),
            float(self.aleatoric_variance_scale),
            calibrator_coefficients(self.calibrator),
            float(self.tau),
        )
