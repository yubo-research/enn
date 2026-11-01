from __future__ import annotations

from dataclasses import dataclass

from enn._rust import ENNParams as _validated_params


@dataclass(frozen=True)
class ENNParams:
    k_num_neighbors: int
    epistemic_variance_scale: float
    aleatoric_variance_scale: float

    def __post_init__(self) -> None:
        _validated_params(
            int(self.k_num_neighbors),
            float(self.epistemic_variance_scale),
            float(self.aleatoric_variance_scale),
        )
