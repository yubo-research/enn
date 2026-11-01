from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from enn.semantic import SemanticInputTransform

from .enn_fit_config import ENNFitConfig
from .enn_x_scaling import validate_tied_dims


def validate_enn_placement(work_dir):
    from enn._rust import validate_enn_placement as rust_validate

    rust_validate(os.fspath(work_dir))


@dataclass(frozen=True)
class ENNSurrogateConfig:
    k: int | None = None
    fit: ENNFitConfig = field(default_factory=ENNFitConfig)
    weighting: bool = False
    tied_dims: tuple = ()
    input_transform: SemanticInputTransform | None = field(default=None, kw_only=True)
    semantic_only: bool = field(default=False, kw_only=True)
    y_bounds: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._check_tied_dims()
        if self.semantic_only and self.input_transform is None:
            raise ValueError("semantic_only requires input_transform")
        if self.y_bounds is not None:
            object.__setattr__(self, "y_bounds", np.asarray(self.y_bounds, dtype=float))

    def _check_tied_dims(self) -> None:
        if not self.tied_dims:
            return
        object.__setattr__(self, "tied_dims", validate_tied_dims(self.tied_dims))

    @property
    def num_fit_samples(self) -> int | None:
        return self.fit.num_fit_samples

    @property
    def num_fit_candidates(self) -> int | None:
        return self.fit.num_fit_candidates
