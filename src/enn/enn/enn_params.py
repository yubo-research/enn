from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ENNParams:
    k: int
    epi_var_scale: float
    ale_homoscedastic_scale: float

    def __post_init__(self) -> None:
        import numpy as np

        k = int(self.k)
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        epi_var_scale = float(self.epi_var_scale)
        if not np.isfinite(epi_var_scale) or epi_var_scale < 0.0:
            raise ValueError(f"epi_var_scale must be >= 0, got {epi_var_scale}")
        ale_scale = float(self.ale_homoscedastic_scale)
        if not np.isfinite(ale_scale) or ale_scale < 0.0:
            raise ValueError(f"ale_homoscedastic_scale must be >= 0, got {ale_scale}")
