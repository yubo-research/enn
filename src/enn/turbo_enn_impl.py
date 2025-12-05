from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

from .base_turbo_impl import BaseTurboImpl
from .turbo_config import TurboConfig


class TurboENNImpl(BaseTurboImpl):
    def __init__(self, config: TurboConfig) -> None:
        super().__init__(config)
        self._enn: Any | None = None
        self._fitted_params: Any | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def needs_tr_list(self) -> bool:
        return True

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
        from .proposal import mk_enn

        k = self._config.k if self._config.k is not None else 10
        self._enn, self._fitted_params, self._y_mean, self._y_std = mk_enn(
            x_obs_list,
            y_obs_list,
            yvar_obs_list=yvar_obs_list,
            k=k,
            num_fit_samples=self._config.num_fit_samples,
            rng=rng,
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
        num_dim: int,
        rng: Generator,
        fallback_fn: Callable[[np.ndarray, int], np.ndarray],
        from_unit_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        import numpy as np

        from .enn_params import ENNParams

        acq_type = self._config.acq_type
        k = self._config.k
        var_scale = self._config.var_scale

        if self._enn is None:
            return fallback_fn(x_cand, num_arms)

        if self._fitted_params is not None:
            params = self._fitted_params
        else:
            k_val = k if k is not None else 10
            params = ENNParams(k=k_val, var_scale=var_scale)

        posterior = self._enn.posterior(x_cand, params=params)
        mu = posterior.mu[:, 0]
        se = posterior.se[:, 0]

        if acq_type == "pareto":
            from .enn_util import arms_from_pareto_fronts

            x_arms = arms_from_pareto_fronts(x_cand, mu, se, num_arms, rng)
        elif acq_type == "ucb":
            beta = 2.0
            scores = mu + beta * se
            shuffled_indices = rng.permutation(len(scores))
            shuffled_scores = scores[shuffled_indices]
            top_k_in_shuffled = np.argpartition(-shuffled_scores, num_arms - 1)[
                :num_arms
            ]
            idx = shuffled_indices[top_k_in_shuffled]
            x_arms = x_cand[idx]
        elif acq_type == "thompson":
            samples = posterior.sample(num_samples=1, rng=rng)
            scores = samples[:, 0, 0]
            shuffled_indices = rng.permutation(len(scores))
            shuffled_scores = scores[shuffled_indices]
            top_k_in_shuffled = np.argpartition(-shuffled_scores, num_arms - 1)[
                :num_arms
            ]
            idx = shuffled_indices[top_k_in_shuffled]
            x_arms = x_cand[idx]
        else:
            raise ValueError(f"Unknown acq_type: {acq_type}")

        return from_unit_fn(x_arms)

    def estimate_y(self, x_unit: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
        if self._enn is None or self._fitted_params is None:
            return y_observed
        posterior = self._enn.posterior(x_unit, params=self._fitted_params)
        mu_standardized = posterior.mu[:, 0]
        return self._y_mean + self._y_std * mu_standardized
