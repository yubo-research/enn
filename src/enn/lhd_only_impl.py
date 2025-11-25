from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .turbo_mode_impl import TurboModeImpl

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

    from .turbo_config import TurboConfig


class LHDOnlyImpl(TurboModeImpl):
    def select_candidates(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        x_tr_list: list | None,
        y_tr_list: list | None,
        num_dim: int,
        k: int | None,
        var_scale: float,
        gp_num_steps: int,
        gp_y_mean: float,
        gp_y_std: float,
        rng: Generator,
        fallback_fn: Callable[[np.ndarray, int], np.ndarray],
        from_unit_fn: Callable[[np.ndarray], np.ndarray],
        gp_model: Any | None,
        gp_y_mean_fitted: float | None,
        gp_y_std_fitted: float | None,
        config: TurboConfig,
    ) -> tuple[np.ndarray, float, float]:
        from .proposal import select_uniform

        return (
            select_uniform(x_cand, num_arms, num_dim, rng, from_unit_fn),
            gp_y_mean,
            gp_y_std,
        )
