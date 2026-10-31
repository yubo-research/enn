from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from enn.semantic import SemanticInputTransform

from .enn_fit_config import ENNFitConfig
from .enn_index_driver import ENNIndexDriver, index_driver_to_wire
from .enn_x_scaling import validate_tied_dims


def validate_enn_placement(*, index_driver, work_dir, weighting):
    from enn._rust import validate_enn_placement as rust_validate

    rust_validate(
        index_driver_to_wire(index_driver),
        None if work_dir is None else os.fspath(work_dir),
        weighting,
    )


@dataclass(frozen=True)
class ENNSurrogateConfig:
    k: int | None = None
    fit: ENNFitConfig = field(default_factory=ENNFitConfig)
    weighting: bool = False
    tied_dims: tuple = ()
    index_driver: ENNIndexDriver = ENNIndexDriver.FLAT
    work_dir: str | os.PathLike[str] | None = None
    input_transform: SemanticInputTransform | None = field(default=None, kw_only=True)
    semantic_only: bool = field(default=False, kw_only=True)
    y_bounds: np.ndarray | None = None

    def __post_init__(self) -> None:
        validate_enn_placement(
            index_driver=self.index_driver,
            work_dir=self.work_dir,
            weighting=self.weighting,
        )
        self._check_tied_dims()
        if self.semantic_only and self.input_transform is None:
            raise ValueError("semantic_only requires input_transform")
        if self.y_bounds is not None:
            object.__setattr__(self, "y_bounds", np.asarray(self.y_bounds, dtype=float))

    def _check_tied_dims(self) -> None:
        if not self.tied_dims:
            return
        flat = [j for g in self.tied_dims for j in g]
        groups = validate_tied_dims(self.tied_dims, 1 + max([0, *flat]))
        object.__setattr__(self, "tied_dims", groups)

    @property
    def num_fit_samples(self) -> int | None:
        return self.fit.num_fit_samples

    @property
    def num_fit_candidates(self) -> int | None:
        return self.fit.num_fit_candidates
