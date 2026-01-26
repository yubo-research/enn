from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from .turbo_utils import ScalarIncumbentMixin, compute_full_box_bounds_1d

if TYPE_CHECKING:
    import numpy as np
    from .components.incumbent_selector import IncumbentSelector
    from .config.no_tr_config import NoTRConfig


@dataclass
class NoTrustRegion(ScalarIncumbentMixin):
    config: NoTRConfig
    num_dim: int
    incumbent_selector: IncumbentSelector = field(default=None, repr=False)
    length: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        from .components.incumbent_selector import ScalarIncumbentSelector

        if self.incumbent_selector is None:
            self.incumbent_selector = ScalarIncumbentSelector(noise_aware=False)

    @property
    def num_metrics(self) -> int:
        return 1

    def update(self, y_obs: np.ndarray | Any, y_incumbent: np.ndarray | Any) -> None:
        return

    def needs_restart(self) -> bool:
        return False

    def restart(self, rng=None) -> None:
        return

    def validate_request(self, num_arms: int, *, is_fallback: bool = False) -> None:
        pass

    def compute_bounds_1d(
        self,
        x_center: np.ndarray | Any,
        lengthscales: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return compute_full_box_bounds_1d(x_center)

    def get_incumbent_indices(
        self,
        y: np.ndarray | Any,
        rng,
        mu: np.ndarray | None = None,
    ) -> np.ndarray:
        import numpy as np

        return np.array([self.get_incumbent_index(y, rng, mu=mu)])
