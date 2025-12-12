from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

from .turbo_config import TurboConfig


class BaseTurboImpl:
    def __init__(self, config: TurboConfig) -> None:
        self._config = config

    def get_x_center(
        self,
        x_obs_list: list,
        y_obs_list: list,
        rng: Generator,
    ) -> np.ndarray | None:
        import numpy as np

        from .turbo_utils import argmax_random_tie

        y_array = np.asarray(y_obs_list, dtype=float)
        if y_array.size == 0:
            return None
        idx = argmax_random_tie(y_array, rng=rng)
        x_array = np.asarray(x_obs_list, dtype=float)
        return x_array[idx]

    def needs_tr_list(self) -> bool:
        return False

    def create_trust_region(self, num_dim: int, num_arms: int) -> Any:
        from .turbo_trust_region import TurboTrustRegion

        return TurboTrustRegion(num_dim=num_dim, num_arms=num_arms)

    def try_early_ask(
        self,
        num_arms: int,
        x_obs_list: list,
        draw_initial_fn: Callable[[int], np.ndarray],
        get_init_lhd_points_fn: Callable[[int], np.ndarray | None],
    ) -> np.ndarray | None:
        return None

    def handle_restart(
        self,
        x_obs_list: list,
        y_obs_list: list,
        yvar_obs_list: list,
        init_idx: int,
        num_init: int,
    ) -> tuple[bool, int]:
        return False, init_idx

    def prepare_ask(
        self,
        x_obs_list: list,
        y_obs_list: list,
        yvar_obs_list: list,
        num_dim: int,
        gp_num_steps: int,
        rng: Any | None = None,
    ) -> tuple[Any, float | None, float | None, np.ndarray | None]:
        return None, None, None, None

    def select_candidates(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        num_dim: int,
        rng: Generator,
        fallback_fn: Callable[[np.ndarray, int], np.ndarray],
        from_unit_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement select_candidates")

    def update_trust_region(
        self,
        tr_state: Any,
        y_obs_list: list,
        x_center: np.ndarray | None = None,
        k: int | None = None,
    ) -> None:
        import numpy as np

        y_obs_array = np.asarray(y_obs_list, dtype=float)
        tr_state.update(y_obs_array)

    def estimate_y(self, x_unit: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
        return y_observed

    def get_mu_sigma(self, x_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        return None
