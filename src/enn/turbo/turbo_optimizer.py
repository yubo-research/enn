from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .proposal import select_uniform
from .turbo_config import TurboConfig
from .turbo_utils import from_unit, latin_hypercube, to_unit


@dataclass(frozen=True)
class Telemetry:
    dt_fit: float
    dt_sel: float


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
        self._yvar_obs_list: list = []
        match mode:
            case TurboMode.TURBO_ONE:
                from .turbo_one_impl import TurboOneImpl

                self._mode_impl: TurboModeImpl = TurboOneImpl(config)
            case TurboMode.TURBO_ZERO:
                from .turbo_zero_impl import TurboZeroImpl

                self._mode_impl = TurboZeroImpl(config)
            case TurboMode.TURBO_ENN:
                from .turbo_enn_impl import TurboENNImpl

                self._mode_impl = TurboENNImpl(config)
            case TurboMode.LHD_ONLY:
                from .lhd_only_impl import LHDOnlyImpl

                self._mode_impl = LHDOnlyImpl(config)
            case _:
                raise ValueError(f"Unknown mode: {mode}")
        self._tr_state: Any | None = None
        self._gp_num_steps: int = 50
        if config.k is not None:
            k_val = int(config.k)
            if k_val < 3:
                raise ValueError(f"k must be >= 3, got {k_val}")
            self._k = k_val
        else:
            self._k = None
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
        if config.local_only:
            center = 0.5 * (self._bounds[:, 0] + self._bounds[:, 1])
            self._init_lhd = center.reshape(1, -1)
            self._num_init = 1
        else:
            self._init_lhd = from_unit(
                latin_hypercube(self._num_init, self._num_dim, rng=self._rng),
                self._bounds,
            )
        self._init_idx = 0
        self._dt_fit: float = 0.0
        self._dt_sel: float = 0.0
        self._local_only = config.local_only

    @property
    def tr_obs_count(self) -> int:
        return len(self._y_obs_list)

    @property
    def best_tr_value(self) -> float | None:
        import numpy as np

        if len(self._y_obs_list) == 0:
            return None
        return float(np.max(self._y_obs_list))

    @property
    def tr_length(self) -> float | None:
        if self._tr_state is None:
            return None
        return float(self._tr_state.length)

    def telemetry(self) -> Telemetry:
        return Telemetry(dt_fit=self._dt_fit, dt_sel=self._dt_sel)

    def ask(self, num_arms: int) -> np.ndarray:
        num_arms = int(num_arms)
        if num_arms <= 0:
            raise ValueError(num_arms)
        if self._tr_state is None:
            self._tr_state = self._mode_impl.create_trust_region(
                self._num_dim, num_arms
            )
            if self._local_only:
                self._tr_state.length_max = 0.1
                self._tr_state.length = min(self._tr_state.length, 0.1)
                self._tr_state.length_init = min(self._tr_state.length_init, 0.1)
        early_result = self._mode_impl.try_early_ask(
            num_arms,
            self._x_obs_list,
            self._draw_initial,
            self._get_init_lhd_points,
        )
        if early_result is not None:
            self._dt_fit = 0.0
            self._dt_sel = 0.0
            return early_result
        if self._init_idx < self._num_init:
            if len(self._x_obs_list) == 0:
                fallback_fn = None
            else:

                def fallback_fn(n: int) -> np.ndarray:
                    return self._ask_normal(n, is_fallback=True)

            self._dt_fit = 0.0
            self._dt_sel = 0.0
            return self._get_init_lhd_points(num_arms, fallback_fn=fallback_fn)
        if len(self._x_obs_list) == 0:
            self._dt_fit = 0.0
            self._dt_sel = 0.0
            return self._draw_initial(num_arms)
        return self._ask_normal(num_arms)

    def _ask_normal(self, num_arms: int, *, is_fallback: bool = False) -> np.ndarray:
        import numpy as np

        if self._tr_state.needs_restart():
            self._tr_state.restart()
            should_reset_init, new_init_idx = self._mode_impl.handle_restart(
                self._x_obs_list,
                self._y_obs_list,
                self._yvar_obs_list,
                self._init_idx,
                self._num_init,
            )
            if should_reset_init:
                self._init_idx = new_init_idx
                self._init_lhd = from_unit(
                    latin_hypercube(self._num_init, self._num_dim, rng=self._rng),
                    self._bounds,
                )
                return self._get_init_lhd_points(num_arms)

        def from_unit_fn(x):
            return from_unit(x, self._bounds)

        if self._mode_impl.needs_tr_list() and len(self._x_obs_list) == 0:
            return self._get_init_lhd_points(num_arms)

        import time

        t0_fit = time.perf_counter()
        _gp_model, _gp_y_mean_fitted, _gp_y_std_fitted, weights = (
            self._mode_impl.prepare_ask(
                self._x_obs_list,
                self._y_obs_list,
                self._yvar_obs_list,
                self._num_dim,
                self._gp_num_steps,
                rng=self._rng,
            )
        )
        self._dt_fit = time.perf_counter() - t0_fit

        x_center = self._mode_impl.get_x_center(
            self._x_obs_list, self._y_obs_list, self._rng
        )
        if x_center is None:
            if len(self._y_obs_list) == 0:
                raise RuntimeError("no observations")
            x_center = np.full(self._num_dim, 0.5)

        x_cand = self._tr_state.generate_candidates(
            x_center,
            weights,
            self._num_candidates,
            self._rng,
            self._sobol_engine,
        )

        def fallback_fn(x, n):
            return select_uniform(x, n, self._num_dim, self._rng, from_unit_fn)

        self._tr_state.validate_request(num_arms, is_fallback=is_fallback)

        t0_sel = time.perf_counter()
        selected = self._mode_impl.select_candidates(
            x_cand,
            num_arms,
            self._num_dim,
            self._rng,
            fallback_fn,
            from_unit_fn,
        )
        self._dt_sel = time.perf_counter() - t0_sel

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
        if incumbent_idx not in indices:
            raise RuntimeError("Incumbent must be included in trimmed list")
        x_array = np.asarray(self._x_obs_list, dtype=float)
        incumbent_value = y_array[incumbent_idx]
        self._x_obs_list = x_array[indices].tolist()
        self._y_obs_list = y_array[indices].tolist()
        if len(self._yvar_obs_list) == len(y_array):
            yvar_array = np.asarray(self._yvar_obs_list, dtype=float)
            self._yvar_obs_list = yvar_array[indices].tolist()
        y_trimmed = np.asarray(self._y_obs_list, dtype=float)
        if not np.any(np.abs(y_trimmed - incumbent_value) < 1e-10):
            raise RuntimeError("Incumbent value must be preserved in trimmed list")

    def tell(
        self,
        x: np.ndarray | Any,
        y: np.ndarray | Any,
        y_var: np.ndarray | Any | None = None,
    ) -> np.ndarray:
        import numpy as np

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        if y_var is not None:
            y_var = np.asarray(y_var, dtype=float)
            if y_var.shape != y.shape:
                raise ValueError((y.shape, y_var.shape))
        if x.shape[0] == 0:
            return np.array([], dtype=float)
        x_unit = to_unit(x, self._bounds)
        y_estimate = self._mode_impl.estimate_y(x_unit, y)
        self._x_obs_list.extend(x_unit.tolist())
        self._y_obs_list.extend(y.tolist())
        if y_var is not None:
            self._yvar_obs_list.extend(y_var.tolist())
        if self._trailing_obs is not None:
            self._trim_trailing_obs()
        self._mode_impl.update_trust_region(self._tr_state, self._y_obs_list)
        return y_estimate

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
