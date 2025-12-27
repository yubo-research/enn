from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

from .turbo_config import TurboZeroConfig


class TurboZeroImpl:
    def __init__(self, config: TurboZeroConfig) -> None:
        self._config = config

    @property
    def always_clears_on_restart(self) -> bool:
        return False

    def get_x_center(
        self,
        x_obs_list: list,
        y_obs_list: list,
        rng: Generator,
        tr_state: Any = None,
    ) -> np.ndarray | None:
        from .impl_helpers import get_x_center_fallback

        return get_x_center_fallback(
            self._config, x_obs_list, y_obs_list, rng, tr_state
        )

    def needs_tr_list(self) -> bool:
        return False

    def try_early_ask(
        self,
        num_arms: int,
        x_obs_list: list,
        draw_initial_fn: Callable[[int], np.ndarray],
        get_init_lhd_points_fn: Callable[[int], np.ndarray],
    ) -> np.ndarray | None:
        return None

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
        tr_state: object | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        from .proposal import select_uniform

        return select_uniform(x_cand, num_arms, num_dim, rng, from_unit_fn)

    def estimate_y(self, x_unit: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
        from .impl_helpers import estimate_y_passthrough

        return estimate_y_passthrough(y_observed)

    def get_mu_sigma(self, x_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        return None
