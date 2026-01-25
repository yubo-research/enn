from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..components.protocols import AcquisitionOptimizer


@dataclass(frozen=True)
class UCBAcquisitionConfig:
    def build(self) -> AcquisitionOptimizer:
        from ..components.acquisition import UCBAcqOptimizer

        return UCBAcqOptimizer()
