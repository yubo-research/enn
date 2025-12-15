from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine


@dataclass
class EpsTrustRegion:
    num_dim: int
    num_arms: int
    eps_tr: float = 0.1
    length: float = 1.0
    length_min: float = 0.1
    _lb: np.ndarray | Any | None = None
    _ub: np.ndarray | Any | None = None
    _center: np.ndarray | Any | None = None

    def __post_init__(self) -> None:
        import numpy as np

        if int(self.num_dim) <= 0:
            raise ValueError(self.num_dim)
        if int(self.num_arms) <= 0:
            raise ValueError(self.num_arms)
        eps_tr = float(self.eps_tr)
        if eps_tr < 0.0 or eps_tr > 1.0:
            raise ValueError(f"eps_tr must be in [0, 1], got {eps_tr}")
        if float(self.length_min) <= 0.0:
            raise ValueError(self.length_min)
        if not np.isfinite(float(self.length_min)):
            raise ValueError(self.length_min)

    def update(self, values: np.ndarray | Any) -> None:
        return

    def needs_restart(self) -> bool:
        return False

    def restart(self) -> None:
        return

    def validate_request(self, num_arms: int, *, is_fallback: bool = False) -> None:
        if is_fallback:
            if num_arms > self.num_arms:
                raise ValueError(
                    f"num_arms {num_arms} > configured num_arms {self.num_arms}"
                )
        else:
            if num_arms != self.num_arms:
                raise ValueError(
                    f"num_arms {num_arms} != configured num_arms {self.num_arms}"
                )

    def update_xy(
        self, x_obs: np.ndarray | Any, y_obs: np.ndarray | Any, *, k: int | None = None
    ) -> None:
        import numpy as np

        x_obs = np.asarray(x_obs, dtype=float)
        y_obs = np.asarray(y_obs, dtype=float)
        if x_obs.ndim != 2 or x_obs.shape[1] != self.num_dim:
            raise ValueError(x_obs.shape)
        if y_obs.ndim != 1 or y_obs.shape[0] != x_obs.shape[0]:
            raise ValueError((x_obs.shape, y_obs.shape))
        if x_obs.shape[0] == 0:
            self._lb = None
            self._ub = None
            self._center = None
            self.length = 1.0
            return

        k_val = int(k) if k is not None else 10
        if k_val <= 0:
            raise ValueError(k_val)
        num_top = min(k_val, y_obs.size)
        top_idx = np.argpartition(-y_obs, num_top - 1)[:num_top]
        x_top = x_obs[top_idx]
        lb = np.min(x_top, axis=0)
        ub = np.max(x_top, axis=0)
        lb = np.clip(lb, 0.0, 1.0)
        ub = np.clip(ub, 0.0, 1.0)
        center = 0.5 * (lb + ub)
        self._lb = lb
        self._ub = ub
        self._center = center
        self.length = max(self.length_min, 1.0 / (1.0 + float(x_obs.shape[0])))

    def _compute_full_bounds_1d(
        self, x_center: np.ndarray | Any
    ) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        lb = np.zeros_like(x_center, dtype=float)
        ub = np.ones_like(x_center, dtype=float)
        return lb, ub

    def generate_candidates(
        self,
        x_center: np.ndarray,
        lengthscales: np.ndarray | None,
        num_candidates: int,
        rng: Generator,
        sobol_engine: QMCEngine,
    ) -> np.ndarray:
        from .turbo_utils import generate_raasp_candidates

        if num_candidates <= 0:
            raise ValueError(num_candidates)
        eps_tr = float(self.eps_tr)
        if rng.random() < eps_tr or self._lb is None or self._ub is None:
            lb, ub = self._compute_full_bounds_1d(x_center)
            center = x_center
        else:
            lb = self._lb
            ub = self._ub
            center = self._center
        return generate_raasp_candidates(
            center, lb, ub, num_candidates, rng=rng, sobol_engine=sobol_engine
        )
