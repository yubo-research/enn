from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

from .turbo_config import TurboConfig


class TurboENNImpl:
    def __init__(self, config: TurboConfig) -> None:
        self._config = config
        self._enn: Any | None = None

    def needs_tr_list(self) -> bool:
        return True

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
        init_idx: int,
        num_init: int,
    ) -> tuple[bool, int]:
        x_obs_list.clear()
        y_obs_list.clear()
        return True, 0

    def prepare_ask(
        self,
        x_obs_list: list,
        y_obs_list: list,
        num_dim: int,
        gp_num_steps: int,
    ) -> tuple[Any, float | None, float | None, np.ndarray | None]:
        from .proposal import mk_enn

        self._enn = mk_enn(
            x_obs_list,
            y_obs_list,
        )
        return None, None, None, None

    def neighbors(
        self,
        x: np.ndarray,
        k: int,
        *,
        exclude_nearest: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._enn is None:
            return []
        return self._enn.neighbors(x, k, exclude_nearest=exclude_nearest)

    def select_candidates(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        x_obs_list: list,
        y_obs_list: list,
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
    ) -> tuple[np.ndarray, float, float]:
        from .proposal import select_enn_pareto

        return (
            select_enn_pareto(
                x_cand,
                num_arms,
                x_obs_list,
                y_obs_list,
                k,
                var_scale,
                rng,
                fallback_fn,
                from_unit_fn,
                enn_model=self._enn,
            ),
            gp_y_mean,
            gp_y_std,
        )

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
