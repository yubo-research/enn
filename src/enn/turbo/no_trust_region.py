from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from .config.no_tr_config import NoTRConfig


@dataclass
class NoTrustRegion:
    config: NoTRConfig
    num_dim: int
    length: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        from .components.incumbent_selector import ScalarIncumbentSelector

        self.incumbent_selector = ScalarIncumbentSelector(noise_aware=True)

    @property
    def num_metrics(self) -> int:
        """No trust region defaults to single-objective."""
        return 1

    def update(self, values: np.ndarray | Any) -> None:
        return

    def needs_restart(self) -> bool:
        return False

    def restart(self, rng=None) -> None:  # noqa: ARG002
        return

    def validate_request(self, num_arms: int, *, is_fallback: bool = False) -> None:  # noqa: ARG002
        pass  # NoTrustRegion has no constraints on num_arms

    def compute_bounds_1d(
        self,
        x_center: np.ndarray | Any,
        lengthscales: np.ndarray | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        from .tr_helpers import compute_full_box_bounds_1d

        return compute_full_box_bounds_1d(x_center)

    def get_incumbent_indices(
        self,
        y: np.ndarray | Any,
        rng,
        mu: np.ndarray | None = None,
    ) -> np.ndarray:
        from .tr_helpers import get_single_incumbent_index

        return get_single_incumbent_index(self.incumbent_selector, y, rng, mu)
