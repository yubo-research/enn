from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


class TurboModeImpl(Protocol):
    def get_x_center(
        self,
        x_obs_list: list,
        y_obs_list: list,
        rng: Generator,
    ) -> np.ndarray | None: ...

    def needs_tr_list(self) -> bool: ...

    def create_trust_region(self, num_dim: int, num_arms: int) -> Any: ...

    def try_early_ask(
        self,
        num_arms: int,
        x_obs_list: list,
        draw_initial_fn: Callable[[int], np.ndarray],
        get_init_lhd_points_fn: Callable[[int], np.ndarray | None],
    ) -> np.ndarray | None: ...

    def handle_restart(
        self,
        x_obs_list: list,
        y_obs_list: list,
        yvar_obs_list: list,
        init_idx: int,
        num_init: int,
    ) -> tuple[bool, int]: ...

    def prepare_ask(
        self,
        x_obs_list: list,
        y_obs_list: list,
        yvar_obs_list: list,
        num_dim: int,
        gp_num_steps: int,
        rng: Generator | Any | None = None,
    ) -> tuple[Any, float | None, float | None, np.ndarray | None]: ...

    def select_candidates(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        num_dim: int,
        rng: Generator,
        fallback_fn: Callable[[np.ndarray, int], np.ndarray],
        from_unit_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray: ...

    def update_trust_region(
        self,
        tr_state: Any,
        y_obs_list: list,
        x_center: np.ndarray | None = None,
        k: int | None = None,
    ) -> None: ...

    def estimate_y(self, x_unit: np.ndarray, y_observed: np.ndarray) -> np.ndarray: ...
