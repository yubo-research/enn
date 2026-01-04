from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass
class NoTrustRegion:
    num_dim: int
    num_arms: int
    length: float = 1.0

    def update(self, values: np.ndarray | Any) -> None:
        return

    def needs_restart(self) -> bool:
        return False

    def restart(self) -> None:
        return

    def validate_request(self, num_arms: int, *, is_fallback: bool = False) -> None:
        from .tr_helpers import validate_trust_region_request

        validate_trust_region_request(num_arms, self.num_arms, is_fallback=is_fallback)

    def compute_bounds_1d(
        self,
        x_center: np.ndarray | Any,
        lengthscales: np.ndarray | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        from .tr_helpers import compute_full_box_bounds_1d

        return compute_full_box_bounds_1d(x_center)
