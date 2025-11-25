from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

    from .turbo_config import TurboConfig


class TurboOneImpl:
    def needs_tr_list(self) -> bool:
        return True

    def create_trust_region(
        self, num_dim: int, num_arms: int, config: TurboConfig
    ) -> Any:
        from .turbo_trust_region import TurboTrustRegion

        return TurboTrustRegion(num_dim=num_dim, num_arms=num_arms)

    def try_early_ask(
        self,
        num_arms: int,
        x_obs_list: list,
        draw_initial_fn: Callable[[int], np.ndarray],
        get_init_lhd_points_fn: Callable[[int], np.ndarray | None],
    ) -> np.ndarray | None:
        if len(x_obs_list) == 0:
            return get_init_lhd_points_fn(num_arms)
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
        import numpy as np

        from .turbo_utils import fit_gp

        if len(x_obs_list) == 0:
            return None, None, None, None
        gp_model, _likelihood, gp_y_mean_fitted, gp_y_std_fitted = fit_gp(
            x_obs_list,
            y_obs_list,
            num_dim,
            num_steps=gp_num_steps,
        )
        weights = None
        if gp_model is not None:
            weights = (
                gp_model.covar_module.base_kernel.lengthscale.cpu()
                .detach()
                .numpy()
                .ravel()
            )
            weights = weights / weights.mean()
            weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
        return gp_model, gp_y_mean_fitted, gp_y_std_fitted, weights

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
        config: TurboConfig,
    ) -> tuple[np.ndarray, float, float]:
        from .proposal import select_gp_thompson

        selected, new_gp_y_mean, new_gp_y_std, _ = select_gp_thompson(
            x_cand,
            num_arms,
            x_obs_list,
            y_obs_list,
            num_dim,
            gp_num_steps,
            rng,
            gp_y_mean,
            gp_y_std,
            fallback_fn,
            from_unit_fn,
            model=gp_model,
            new_gp_y_mean=gp_y_mean_fitted,
            new_gp_y_std=gp_y_std_fitted,
        )
        return selected, new_gp_y_mean, new_gp_y_std

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
