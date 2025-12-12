from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ENNParams:
    k: int
    epi_var_scale: float
    ale_homoscedastic_scale: float
