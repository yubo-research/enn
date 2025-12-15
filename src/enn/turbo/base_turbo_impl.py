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
        tr_state: Any = None,
    ) -> np.ndarray | None:
        import numpy as np

        from .turbo_utils import argmax_random_tie

        y_array = np.asarray(y_obs_list, dtype=float)
        if y_array.size == 0:
            return None
        x_array = np.asarray(x_obs_list, dtype=float)

        # For morbo: scalarize raw y observations
        if self._config.tr_type == "morbo" and tr_state is not None:
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, tr_state.num_metrics)
            scalarized = tr_state.scalarize(y_array, clip=True)
            idx = argmax_random_tie(scalarized, rng=rng)
        else:
            idx = argmax_random_tie(y_array, rng=rng)

        return x_array[idx]

    def needs_tr_list(self) -> bool:
        return False

    def create_trust_region(
        self,
        num_dim: int,
        num_arms: int,
        rng: Generator,
        num_metrics: int | None = None,
    ) -> Any:
        if self._config.tr_type == "none":
            from .no_trust_region import NoTrustRegion

            return NoTrustRegion(num_dim=num_dim, num_arms=num_arms)
        elif self._config.tr_type == "turbo":
            from .turbo_trust_region import TurboTrustRegion

            return TurboTrustRegion(num_dim=num_dim, num_arms=num_arms)
        elif self._config.tr_type == "morbo":
            from .morbo_trust_region import MorboTrustRegion

            effective_num_metrics = num_metrics or self._config.num_metrics
            if effective_num_metrics is None:
                raise ValueError("num_metrics required for tr_type='morbo'")
            return MorboTrustRegion(
                num_dim=num_dim,
                num_arms=num_arms,
                num_metrics=effective_num_metrics,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown tr_type: {self._config.tr_type!r}")

    def try_early_ask(
        self,
        num_arms: int,
        x_obs_list: list,
        draw_initial_fn: Callable[[int], np.ndarray],
        get_init_lhd_points_fn: Callable[[int], np.ndarray],
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
        if self._config.tr_type == "morbo":
            x_obs_list.clear()
            y_obs_list.clear()
            yvar_obs_list.clear()
            return True, 0
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
        tr_state: Any = None,
    ) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement select_candidates")

    def update_trust_region(
        self,
        tr_state: Any,
        x_obs_list: list,
        y_obs_list: list,
        x_center: np.ndarray | None = None,
        k: int | None = None,
    ) -> None:
        import numpy as np

        x_obs_array = np.asarray(x_obs_list, dtype=float)
        y_obs_array = np.asarray(y_obs_list, dtype=float)
        if hasattr(tr_state, "update_xy"):
            tr_state.update_xy(x_obs_array, y_obs_array, k=k)
        else:
            tr_state.update(y_obs_array)

    def estimate_y(self, x_unit: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
        return y_observed

    def get_mu_sigma(self, x_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        return None
