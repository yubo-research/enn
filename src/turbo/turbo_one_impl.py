from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

from .base_turbo_impl import BaseTurboImpl
from .turbo_config import TurboConfig
from .turbo_utils import gp_thompson_sample


class TurboOneImpl(BaseTurboImpl):
    def __init__(self, config: TurboConfig) -> None:
        super().__init__(config)
        self._gp_model: Any | None = None
        self._gp_y_mean: float = 0.0
        self._gp_y_std: float = 1.0
        self._fitted_n_obs: int = 0

    def get_x_center(
        self,
        x_obs_list: list,
        y_obs_list: list,
        rng: Generator,
    ) -> np.ndarray | None:
        import numpy as np
        import torch

        from .turbo_utils import argmax_random_tie

        if len(y_obs_list) == 0:
            return None
        if self._gp_model is None:
            return super().get_x_center(x_obs_list, y_obs_list, rng)
        if self._fitted_n_obs != len(x_obs_list):
            raise RuntimeError(
                f"GP fitted on {self._fitted_n_obs} obs but get_x_center called with {len(x_obs_list)}"
            )

        x_array = np.asarray(x_obs_list, dtype=float)
        x_torch = torch.as_tensor(x_array, dtype=torch.float64)
        with torch.no_grad():
            posterior = self._gp_model.posterior(x_torch)
            mu = posterior.mean.cpu().numpy().ravel()

        best_idx = argmax_random_tie(mu, rng=rng)
        return x_array[best_idx]

    def needs_tr_list(self) -> bool:
        return True

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
        yvar_obs_list: list,
        init_idx: int,
        num_init: int,
    ) -> tuple[bool, int]:
        x_obs_list.clear()
        y_obs_list.clear()
        yvar_obs_list.clear()
        return True, 0

    def prepare_ask(
        self,
        x_obs_list: list,
        y_obs_list: list,
        yvar_obs_list: list,
        num_dim: int,
        gp_num_steps: int,
        rng: Any | None = None,
    ) -> tuple[Any, float | None, float | None, np.ndarray | None]:
        import numpy as np

        from .turbo_utils import fit_gp

        if len(x_obs_list) == 0:
            return None, None, None, None
        self._gp_model, _likelihood, gp_y_mean_fitted, gp_y_std_fitted = fit_gp(
            x_obs_list,
            y_obs_list,
            num_dim,
            yvar_obs_list=yvar_obs_list if yvar_obs_list else None,
            num_steps=gp_num_steps,
        )
        self._fitted_n_obs = len(x_obs_list)
        if gp_y_mean_fitted is not None:
            self._gp_y_mean = gp_y_mean_fitted
        if gp_y_std_fitted is not None:
            self._gp_y_std = gp_y_std_fitted
        weights = None
        if self._gp_model is not None:
            weights = (
                self._gp_model.covar_module.base_kernel.lengthscale.cpu()
                .detach()
                .numpy()
                .ravel()
            )
            # First line helps stabilize second line.
            weights = weights / weights.mean()
            weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
        return self._gp_model, gp_y_mean_fitted, gp_y_std_fitted, weights

    def select_candidates(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        num_dim: int,
        rng: Generator,
        fallback_fn: Callable[[np.ndarray, int], np.ndarray],
        from_unit_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        if self._gp_model is None:
            return fallback_fn(x_cand, num_arms)

        idx = gp_thompson_sample(
            self._gp_model,
            x_cand,
            num_arms,
            rng,
            self._gp_y_mean,
            self._gp_y_std,
        )
        return from_unit_fn(x_cand[idx])

    def estimate_y(self, x_unit: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
        import torch

        if self._gp_model is None:
            return y_observed
        x_torch = torch.as_tensor(x_unit, dtype=torch.float64)
        with torch.no_grad():
            posterior = self._gp_model.posterior(x_torch)
            mu = posterior.mean.cpu().numpy().ravel()
        return self._gp_y_mean + self._gp_y_std * mu

    def get_mu_sigma(self, x_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        import torch

        if self._gp_model is None:
            return None
        x_torch = torch.as_tensor(x_unit, dtype=torch.float64)
        with torch.no_grad():
            posterior = self._gp_model.posterior(x_torch)
            mu_std = posterior.mean.cpu().numpy().ravel()
            sigma_std = posterior.variance.cpu().numpy().ravel() ** 0.5
        mu = self._gp_y_mean + self._gp_y_std * mu_std
        sigma = self._gp_y_std * sigma_std
        return mu, sigma
