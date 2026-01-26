from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..components.protocols import AcquisitionOptimizer


@dataclass(frozen=True)
class ParetoAcquisitionConfig:
    def build(self) -> AcquisitionOptimizer:
        from ..components.builder import build_acquisition_optimizer

        return build_acquisition_optimizer(self)
