from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .turbo_mode_impl import TurboModeImpl

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

    from .turbo_config import TurboConfig


class TurboOneImpl(TurboModeImpl):
    def needs_tr_list(self) -> bool:
        return True

    def get_x_center(
        self,
        x_obs_list: list,
        y_obs_list: list,
        x_tr_list: list | None,
        y_tr_list: list | None,
        argmax_random_tie_fn: Callable[..., int],
        rng: Generator,
    ) -> np.ndarray:
        import numpy as np

        if x_tr_list is not None and len(x_tr_list) > 0:
            y_array = np.asarray(y_tr_list, dtype=float)
            if y_array.size == 0:
                raise RuntimeError("no trust-region observations")
            idx = argmax_random_tie_fn(y_array, rng=rng)
            x_array = np.asarray(x_tr_list, dtype=float)
            return x_array[idx]
        y_array = np.asarray(y_obs_list, dtype=float)
        if y_array.size == 0:
            raise RuntimeError("no observations")
        idx = argmax_random_tie_fn(y_array, rng=rng)
        x_array = np.asarray(x_obs_list, dtype=float)
        return x_array[idx]

    def handle_restart(
        self,
        x_tr_list: list | None,
        y_tr_list: list | None,
        init_idx: int,
        num_init: int,
    ) -> tuple[bool, int]:
        if x_tr_list is not None:
            x_tr_list.clear()
            y_tr_list.clear()
            return True, 0
        return False, init_idx

    def prepare_ask(
        self,
        x_tr_list: list | None,
        y_tr_list: list | None,
        num_dim: int,
        gp_num_steps: int,
    ) -> tuple[Any, float | None, float | None, np.ndarray | None]:
        import numpy as np

        from .turbo_utils import fit_gp

        if x_tr_list is None or len(x_tr_list) == 0:
            return None, None, None, None
        gp_model, _likelihood, gp_y_mean_fitted, gp_y_std_fitted = fit_gp(
            x_tr_list,
            y_tr_list,
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
        from .proposal import select_gp_thompson

        if x_tr_list is None:
            x_tr_list = []
        if y_tr_list is None:
            y_tr_list = []
        selected, new_gp_y_mean, new_gp_y_std, _ = select_gp_thompson(
            x_cand,
            num_arms,
            x_tr_list,
            y_tr_list,
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
