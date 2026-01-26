from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..components.protocols import Surrogate


@dataclass(frozen=True)
class GPSurrogateConfig:
    def build(self) -> Surrogate:
        from ..components.builder import build_surrogate

        return build_surrogate(self)
