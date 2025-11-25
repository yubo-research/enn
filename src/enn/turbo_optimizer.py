from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .proposal import select_uniform
from .turbo_config import TurboConfig
from .turbo_utils import argmax_random_tie, from_unit, latin_hypercube, raasp, to_unit

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

    from .turbo_mode import TurboMode
    from .turbo_mode_impl import TurboModeImpl


class TurboOptimizer:
    def __init__(
        self,
        bounds: np.ndarray,
        mode: TurboMode,
        num_arms: int,
        *,
        rng: Generator,
        config: TurboConfig | None = None,
    ) -> None:
        import numpy as np
        from scipy.stats import qmc

        from .turbo_mode import TurboMode

        if config is None:
            config = TurboConfig()
        self._config = config

        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(bounds.shape)
        self._bounds = np.asarray(bounds, dtype=float)
        self._num_dim = self._bounds.shape[0]
        self._mode = mode
        tr_num_arms = int(num_arms)
        if tr_num_arms <= 0:
            raise ValueError(tr_num_arms)
        num_candidates = config.num_candidates
        if num_candidates is None:
            num_candidates = min(5000, 100 * self._num_dim)

        self._num_candidates = int(num_candidates)
        if self._num_candidates <= 0:
            raise ValueError(self._num_candidates)
        self._rng = rng
        sobol_seed = int(self._rng.integers(1_000_000))
        self._sobol_engine = qmc.Sobol(d=self._num_dim, scramble=True, seed=sobol_seed)
        self._x_obs_list: list = []
        self._y_obs_list: list = []
        match mode:
            case TurboMode.TURBO_ONE:
                from .turbo_one_impl import TurboOneImpl

                self._mode_impl: TurboModeImpl = TurboOneImpl()
            case TurboMode.TURBO_ZERO:
                from .turbo_zero_impl import TurboZeroImpl

                self._mode_impl = TurboZeroImpl()
            case TurboMode.TURBO_ENN:
                from .turbo_enn_impl import TurboENNImpl

                self._mode_impl = TurboENNImpl()
            case TurboMode.LHD_ONLY:
                from .lhd_only_impl import LHDOnlyImpl

                self._mode_impl = LHDOnlyImpl()
            case _:
                raise ValueError(f"Unknown mode: {mode}")
        self._tr_state = self._mode_impl.create_trust_region(
            self._num_dim, tr_num_arms, config
        )
        self._gp_y_mean: float = 0.0
        self._gp_y_std: float = 1.0
        self._gp_num_steps: int = 50
        if config.k is not None:
            k_val = int(config.k)
            if k_val < 3:
                raise ValueError(f"k must be >= 3, got {k_val}")
            self._k = k_val
        else:
            self._k = None
        var_scale_val = float(config.var_scale)
        if var_scale_val <= 0.0:
            raise ValueError(f"var_scale must be > 0, got {var_scale_val}")
        self._var_scale = var_scale_val
        if config.trailing_obs is not None:
            trailing_obs_val = int(config.trailing_obs)
            if trailing_obs_val <= 0:
                raise ValueError(f"trailing_obs must be > 0, got {trailing_obs_val}")
            self._trailing_obs = trailing_obs_val
        else:
            self._trailing_obs = None
        num_init = config.num_init
        if num_init is None:
            num_init = 2 * self._num_dim
        num_init_val = int(num_init)
        if num_init_val <= 0:
            raise ValueError(f"num_init must be > 0, got {num_init_val}")
        self._num_init = num_init_val
        self._init_lhd = from_unit(
            latin_hypercube(self._num_init, self._num_dim, rng=self._rng),
            self._bounds,
        )
        self._init_idx = 0

    @property
    def tr_obs_count(self) -> int:
        return len(self._y_obs_list)

    @property
    def best_tr_value(self) -> float | None:
        import numpy as np

        if len(self._y_obs_list) == 0:
            return None
        return float(np.max(self._y_obs_list))

    def ask(self, num_arms: int) -> np.ndarray:
        num_arms = int(num_arms)
        if num_arms <= 0:
            raise ValueError(num_arms)
        early_result = self._mode_impl.try_early_ask(
            num_arms,
            self._x_obs_list,
            self._draw_initial,
            self._get_init_lhd_points,
        )
        if early_result is not None:
            return early_result
        if self._init_idx < self._num_init:
            if len(self._x_obs_list) == 0:
                fallback_fn = None
            else:
                fallback_fn = self._ask_normal
            return self._get_init_lhd_points(num_arms, fallback_fn=fallback_fn)
        if len(self._x_obs_list) == 0:
            return self._draw_initial(num_arms)
        return self._ask_normal(num_arms)

    def _ask_normal(self, num_arms: int) -> np.ndarray:
        if self._tr_state.needs_restart():
            self._tr_state.restart()
            should_reset_init, new_init_idx = self._mode_impl.handle_restart(
                self._x_obs_list, self._y_obs_list, self._init_idx, self._num_init
            )
            if should_reset_init:
                self._init_idx = new_init_idx
                self._init_lhd = from_unit(
                    latin_hypercube(self._num_init, self._num_dim, rng=self._rng),
                    self._bounds,
                )
                return self._get_init_lhd_points(num_arms)

        x_center = self._best_x_from_lists(
            self._x_obs_list, self._y_obs_list, "no observations"
        )

        def from_unit_fn(x):
            return from_unit(x, self._bounds)

        if self._mode_impl.needs_tr_list() and len(self._x_obs_list) == 0:
            return self._get_init_lhd_points(num_arms)

        gp_model, gp_y_mean_fitted, gp_y_std_fitted, weights = (
            self._mode_impl.prepare_ask(
                self._x_obs_list,
                self._y_obs_list,
                self._num_dim,
                self._gp_num_steps,
            )
        )

        lb_local, ub_local = self._tr_state.compute_bounds_1d(x_center, weights)

        x_cand = raasp(
            x_center,
            lb_local,
            ub_local,
            self._num_candidates,
            num_pert=20,
            rng=self._rng,
            sobol_engine=self._sobol_engine,
        )

        def fallback_fn(x, n):
            return select_uniform(x, n, self._num_dim, self._rng, from_unit_fn)

        selected, self._gp_y_mean, self._gp_y_std = self._mode_impl.select_candidates(
            x_cand,
            num_arms,
            self._x_obs_list,
            self._y_obs_list,
            self._num_dim,
            self._k,
            self._var_scale,
            self._gp_num_steps,
            self._gp_y_mean,
            self._gp_y_std,
            self._rng,
            fallback_fn,
            from_unit_fn,
            gp_model,
            gp_y_mean_fitted,
            gp_y_std_fitted,
            self._config,
        )

        self._mode_impl.update_trust_region(
            self._tr_state, self._y_obs_list, x_center=x_center, k=self._k
        )
        return selected

    def _trim_trailing_obs(self) -> None:
        import numpy as np

        from .turbo_utils import argmax_random_tie

        if len(self._x_obs_list) <= self._trailing_obs:
            return
        y_array = np.asarray(self._y_obs_list, dtype=float)
        incumbent_idx = argmax_random_tie(y_array, rng=self._rng)
        num_total = len(self._x_obs_list)
        start_idx = max(0, num_total - self._trailing_obs)
        if incumbent_idx < start_idx:
            indices = np.array(
                [incumbent_idx]
                + list(range(num_total - (self._trailing_obs - 1), num_total)),
                dtype=int,
            )
        else:
            indices = np.arange(start_idx, num_total, dtype=int)
        assert incumbent_idx in indices, "Incumbent must be included in trimmed list"
        x_array = np.asarray(self._x_obs_list, dtype=float)
        incumbent_value = y_array[incumbent_idx]
        self._x_obs_list = x_array[indices].tolist()
        self._y_obs_list = y_array[indices].tolist()
        y_trimmed = np.asarray(self._y_obs_list, dtype=float)
        assert np.any(
            np.abs(y_trimmed - incumbent_value) < 1e-10
        ), "Incumbent value must be preserved in trimmed list"

    def _append_observations(self, x: np.ndarray | Any, y: np.ndarray | Any) -> None:
        import numpy as np

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        if x.shape[0] == 0:
            return
        x_unit = to_unit(x, self._bounds)
        self._x_obs_list.extend(x_unit.tolist())
        self._y_obs_list.extend(y.tolist())
        if self._trailing_obs is not None:
            self._trim_trailing_obs()
        y_obs_array = np.asarray(self._y_obs_list, dtype=float)
        self._tr_state.update(y_obs_array)

    def tell(self, x: np.ndarray | Any, y: np.ndarray | Any) -> None:
        self._append_observations(x, y)

    def _best_x_from_lists(
        self,
        x_list: list[float] | list[list[float]],
        y_list: list[float] | list[list[float]],
        error_msg: str,
    ) -> np.ndarray:
        import numpy as np

        y_array = np.asarray(y_list, dtype=float)
        if y_array.size == 0:
            raise RuntimeError(error_msg)
        idx = argmax_random_tie(y_array, rng=self._rng)
        x_array = np.asarray(x_list, dtype=float)
        return x_array[idx]

    def _draw_initial(self, num_arms: int) -> np.ndarray:
        unit = latin_hypercube(num_arms, self._num_dim, rng=self._rng)
        return from_unit(unit, self._bounds)

    def _get_init_lhd_points(
        self, num_arms: int, fallback_fn: Callable[[int], np.ndarray] | None = None
    ) -> np.ndarray:
        import numpy as np

        remaining_init = self._num_init - self._init_idx
        num_to_return = min(num_arms, remaining_init)
        result = self._init_lhd[self._init_idx : self._init_idx + num_to_return]
        self._init_idx += num_to_return
        if num_to_return < num_arms:
            num_remaining = num_arms - num_to_return
            if fallback_fn is not None:
                result = np.vstack([result, fallback_fn(num_remaining)])
            else:
                result = np.vstack([result, self._draw_initial(num_remaining)])
        return result
