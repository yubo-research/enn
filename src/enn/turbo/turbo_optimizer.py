from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .proposal import select_uniform
from .turbo_mode_registry import make_impl, validate_config
from .turbo_utils import (
    Telemetry,
    from_unit,
    latin_hypercube,
    sobol_seed_for_state,
    to_unit,
    trim_trailing_observations,
    validate_tell_inputs,
)

if TYPE_CHECKING:
    from numpy.random import Generator
    from .turbo_config import TurboConfig
    from .turbo_mode import TurboMode


class TurboOptimizer:
    def __init__(
        self,
        bounds: np.ndarray,
        mode: TurboMode,
        *,
        rng: Generator,
        config: TurboConfig | None = None,
    ) -> None:
        config = validate_config(mode, config)
        self._config = config

        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(bounds.shape)
        self._bounds = bounds
        self._num_dim = self._bounds.shape[0]
        self._mode = mode
        num_candidates = config.num_candidates
        if num_candidates is None:
            num_candidates = min(5000, 100 * self._num_dim)

        self._num_candidates = int(num_candidates)
        if self._num_candidates <= 0:
            raise ValueError(self._num_candidates)
        self._rng = rng
        self._sobol_seed_base = int(self._rng.integers(2**31 - 1))
        self._x_obs_list: list[list[float]] = []
        self._y_obs_list: list[float] | list[list[float]] = []
        self._y_tr_list: list[float] | list[list[float]] = []
        self._yvar_obs_list: list[float] | list[list[float]] = []
        self._expects_yvar: bool | None = None
        self._mode_impl: Any = make_impl(mode, config)
        self._tr_state: Any | None = None
        self._gp_num_steps: int = 50
        self._k = None if config.k is None else int(config.k)
        if self._k is not None and self._k < 3:
            raise ValueError(f"k must be >= 3, got {self._k}")
        self._trailing_obs = (
            None if config.trailing_obs is None else int(config.trailing_obs)
        )
        if self._trailing_obs is not None and self._trailing_obs <= 0:
            raise ValueError(f"trailing_obs must be > 0, got {self._trailing_obs}")
        self._num_init = int(
            config.num_init if config.num_init is not None else 2 * self._num_dim
        )
        if self._num_init <= 0:
            raise ValueError(f"num_init must be > 0, got {self._num_init}")
        self._init_lhd = from_unit(
            latin_hypercube(self._num_init, self._num_dim, rng=self._rng),
            self._bounds,
        )
        self._init_idx = 0
        self._dt_fit: float = 0.0
        self._dt_sel: float = 0.0

    @property
    def tr_obs_count(self) -> int:
        return len(self._y_obs_list)

    @property
    def tr_length(self) -> float | None:
        if self._tr_state is None:
            return None
        if not hasattr(self._tr_state, "length"):
            return None
        return float(self._tr_state.length)

    def telemetry(self) -> Telemetry:
        return Telemetry(dt_fit=self._dt_fit, dt_sel=self._dt_sel)

    def _reset_timing(self) -> None:
        self._dt_fit, self._dt_sel = 0.0, 0.0

    def ask(self, num_arms: int) -> np.ndarray:
        num_arms = int(num_arms)
        if num_arms <= 0:
            raise ValueError(num_arms)
        if self._tr_state is not None:
            self._tr_state.validate_request(num_arms)
        early = self._mode_impl.try_early_ask(
            num_arms,
            self._x_obs_list,
            self._draw_initial,
            self._get_init_lhd_points,
        )
        if early is not None:
            self._reset_timing()
            return early
        if self._init_idx < self._num_init:

            def fallback(n):
                return self._ask_normal(n, is_fallback=True)

            self._reset_timing()
            return self._get_init_lhd_points(
                num_arms, fallback_fn=fallback if self._x_obs_list else None
            )
        if not self._x_obs_list:
            self._reset_timing()
            return self._draw_initial(num_arms)
        return self._ask_normal(num_arms)

    def _handle_tr_restart(self, num_arms: int) -> np.ndarray | None:
        if not self._tr_state.needs_restart():
            return None
        self._tr_state.restart()
        should_reset_init, new_init_idx = self._mode_impl.handle_restart(
            self._x_obs_list,
            self._y_obs_list,
            self._yvar_obs_list,
            self._init_idx,
            self._num_init,
        )
        if not should_reset_init:
            return None
        self._y_tr_list = []
        self._init_idx = new_init_idx
        self._init_lhd = from_unit(
            latin_hypercube(self._num_init, self._num_dim, rng=self._rng),
            self._bounds,
        )
        return self._get_init_lhd_points(num_arms)

    def _ask_normal(self, num_arms: int, *, is_fallback: bool = False) -> np.ndarray:
        import time
        from scipy.stats import qmc

        if self._tr_state is None:
            return self._draw_initial(num_arms)
        restart_result = self._handle_tr_restart(num_arms)
        if restart_result is not None:
            return restart_result
        if self._mode_impl.needs_tr_list() and not self._x_obs_list:
            return self._get_init_lhd_points(num_arms)

        t0_fit = time.perf_counter()
        _, _, _, lengthscales = self._mode_impl.prepare_ask(
            self._x_obs_list,
            self._y_obs_list,
            self._yvar_obs_list,
            self._num_dim,
            self._gp_num_steps,
            rng=self._rng,
        )
        self._dt_fit = time.perf_counter() - t0_fit

        x_center = self._mode_impl.get_x_center(
            self._x_obs_list,
            self._y_obs_list,
            self._rng,
            self._tr_state,
        )
        if x_center is None:
            if not self._y_obs_list:
                raise RuntimeError("no observations")
            x_center = np.full(self._num_dim, 0.5)

        sobol_seed = sobol_seed_for_state(
            self._sobol_seed_base, n_obs=len(self._x_obs_list), num_arms=num_arms
        )
        sobol_engine = qmc.Sobol(d=self._num_dim, scramble=True, seed=sobol_seed)
        x_cand = self._tr_state.generate_candidates(
            x_center,
            lengthscales,
            self._num_candidates,
            self._rng,
            sobol_engine,
        )

        def from_unit_fn(x):
            return from_unit(x, self._bounds)

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
            tr_state=self._tr_state,
        )
        self._dt_sel = time.perf_counter() - t0_sel
        return selected

    def _trim_trailing_obs(self) -> None:
        y_tr_array = np.asarray(self._y_tr_list, dtype=float)
        incumbent_indices = self._tr_state.get_incumbent_indices(y_tr_array, self._rng)
        self._x_obs_list, self._y_obs_list, self._y_tr_list, self._yvar_obs_list = (
            trim_trailing_observations(
                self._x_obs_list,
                self._y_obs_list,
                self._y_tr_list,
                self._yvar_obs_list,
                trailing_obs=self._trailing_obs,
                incumbent_indices=incumbent_indices,
            )
        )

    def tell(
        self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None
    ) -> np.ndarray:
        x, y, y_var, num_metrics = validate_tell_inputs(x, y, y_var, self._num_dim)
        if self._config.tr_type != "morbo" and num_metrics != 1:
            raise ValueError(
                f"Single-objective requires num_metrics=1, got {num_metrics}"
            )
        if self._tr_state is None:
            self._tr_state = self._mode_impl.create_trust_region(
                self._num_dim, x.shape[0], self._rng, num_metrics=num_metrics
            )
        cfg_nm = self._config.num_metrics
        if cfg_nm is not None and num_metrics != cfg_nm:
            raise ValueError(f"y has {num_metrics} metrics but expected {cfg_nm}")
        if self._expects_yvar is None:
            self._expects_yvar = y_var is not None
        if (y_var is not None) != bool(self._expects_yvar):
            raise ValueError(
                f"y_var must be {'provided' if self._expects_yvar else 'omitted'} on every tell()"
            )
        if x.shape[0] == 0:
            return (
                np.array([], dtype=float)
                if num_metrics == 1
                else np.empty((0, num_metrics), dtype=float)
            )

        x_unit = to_unit(x, self._bounds)
        self._x_obs_list.extend(x_unit.tolist())
        self._y_obs_list.extend(y.tolist())
        if y_var is not None:
            self._yvar_obs_list.extend(y_var.tolist())

        x_all, y_all = (
            np.asarray(self._x_obs_list, dtype=float),
            np.asarray(self._y_obs_list, dtype=float),
        )
        self._mode_impl.prepare_ask(
            self._x_obs_list,
            self._y_obs_list,
            self._yvar_obs_list,
            self._num_dim,
            0,
            rng=self._rng,
        )
        mu_all = np.asarray(self._mode_impl.estimate_y(x_all, y_all), dtype=float)
        y_estimate = np.asarray(self._mode_impl.estimate_y(x_unit, y), dtype=float)
        self._y_tr_list = mu_all.tolist()

        if self._trailing_obs is not None:
            self._trim_trailing_obs()

        prev_n = int(getattr(self._tr_state, "prev_num_obs", 0))
        if (
            prev_n > 0
            and prev_n <= len(self._y_tr_list)
            and hasattr(self._tr_state, "best_value")
        ):
            y_tr = np.asarray(self._y_tr_list, dtype=float)
            self._tr_state.best_value = float(
                np.max((y_tr[:, 0] if y_tr.ndim == 2 else y_tr)[:prev_n])
            )

        self._tr_state.update(np.asarray(self._y_tr_list, dtype=float))
        return y_estimate

    def _draw_initial(self, num_arms: int) -> np.ndarray:
        unit = latin_hypercube(num_arms, self._num_dim, rng=self._rng)
        return from_unit(unit, self._bounds)

    def _get_init_lhd_points(
        self, num_arms: int, fallback_fn: Callable[[int], np.ndarray] | None = None
    ) -> np.ndarray:
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
