from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .proposal import select_uniform
from .turbo_config import (
    LHDOnlyConfig,
    TurboConfig,
    TurboENNConfig,
    TurboOneConfig,
    TurboZeroConfig,
)
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

        from .turbo_mode import TurboMode

        mode_registry: dict[TurboMode, tuple[type, type]] = {
            TurboMode.TURBO_ONE: (TurboOneConfig, self._make_turbo_one_impl),
            TurboMode.TURBO_ZERO: (TurboZeroConfig, self._make_turbo_zero_impl),
            TurboMode.TURBO_ENN: (TurboENNConfig, self._make_turbo_enn_impl),
            TurboMode.LHD_ONLY: (LHDOnlyConfig, self._make_lhd_only_impl),
        }

        if mode not in mode_registry:
            raise ValueError(f"Unknown mode: {mode}")

        config_class, _ = mode_registry[mode]
        if config is None:
            config = config_class()
        elif not isinstance(config, config_class):
            raise ValueError(
                f"mode={mode} requires {config_class.__name__}, got {type(config).__name__}"
            )
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
        _, impl_factory = mode_registry[mode]
        self._mode_impl: TurboModeImpl = impl_factory(config)
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
        self._init_lhd = from_unit(
            latin_hypercube(self._num_init, self._num_dim, rng=self._rng),
            self._bounds,
        )
        self._init_idx = 0
        self._dt_fit: float = 0.0
        self._dt_sel: float = 0.0

    def _sobol_seed_for_state(self, *, n_obs: int, num_arms: int) -> int:
        mask64 = (1 << 64) - 1

        x = int(self._sobol_seed_base) & mask64
        x ^= (int(n_obs) + 1) * 0x9E3779B97F4A7C15 & mask64
        x ^= (int(num_arms) + 1) * 0xBF58476D1CE4E5B9 & mask64
        x = (x + 0x9E3779B97F4A7C15) & mask64
        z = x
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & mask64
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & mask64
        z = z ^ (z >> 31)
        return int(z & 0xFFFFFFFF)

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

    def ask(self, num_arms: int) -> np.ndarray:
        num_arms = int(num_arms)
        if num_arms <= 0:
            raise ValueError(num_arms)
        if self._tr_state is not None:
            self._tr_state.validate_request(num_arms)
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
        from scipy.stats import qmc

        if self._tr_state is None:
            return self._draw_initial(num_arms)

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
                self._y_tr_list = []
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
        _gp_model, _gp_y_mean_fitted, _gp_y_std_fitted, lengthscales = (
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
            self._x_obs_list,
            self._y_obs_list,
            self._rng,
            self._tr_state,
        )
        if x_center is None:
            if len(self._y_obs_list) == 0:
                raise RuntimeError("no observations")
            x_center = np.full(self._num_dim, 0.5)

        sobol_seed = self._sobol_seed_for_state(
            n_obs=len(self._x_obs_list),
            num_arms=num_arms,
        )
        sobol_engine = qmc.Sobol(d=self._num_dim, scramble=True, seed=sobol_seed)
        x_cand = self._tr_state.generate_candidates(
            x_center,
            lengthscales,
            self._num_candidates,
            self._rng,
            sobol_engine,
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
            tr_state=self._tr_state,
        )
        self._dt_sel = time.perf_counter() - t0_sel
        return selected

    def _trim_trailing_obs(self) -> None:
        import numpy as np

        if len(self._x_obs_list) <= self._trailing_obs:
            return
        y_tr_array = np.asarray(self._y_tr_list, dtype=float)
        incumbent_indices = self._tr_state.get_incumbent_indices(y_tr_array, self._rng)
        num_total = len(self._x_obs_list)
        start_idx = max(0, num_total - self._trailing_obs)

        recent_indices = set(range(start_idx, num_total))
        keep_indices = set(incumbent_indices.tolist()) | recent_indices

        if len(keep_indices) > self._trailing_obs:
            keep_indices = set(incumbent_indices.tolist())
            remaining_slots = self._trailing_obs - len(keep_indices)
            if remaining_slots > 0:
                recent_non_incumbent = [
                    i for i in range(num_total - 1, -1, -1) if i not in keep_indices
                ][:remaining_slots]
                keep_indices.update(recent_non_incumbent)

        indices = np.array(sorted(keep_indices), dtype=int)

        x_array = np.asarray(self._x_obs_list, dtype=float)
        self._x_obs_list = x_array[indices].tolist()
        y_obs_array = np.asarray(self._y_obs_list, dtype=float)
        self._y_obs_list = y_obs_array[indices].tolist()
        self._y_tr_list = y_tr_array[indices].tolist()
        if len(self._yvar_obs_list) == len(y_obs_array):
            yvar_array = np.asarray(self._yvar_obs_list, dtype=float)
            self._yvar_obs_list = yvar_array[indices].tolist()

    def tell(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_var: np.ndarray | None = None,
    ) -> np.ndarray:
        import numpy as np

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)

        if y.ndim == 2:
            if y.shape[0] != x.shape[0]:
                raise ValueError((x.shape, y.shape))
            num_metrics = y.shape[1]
        elif y.ndim == 1:
            if y.shape[0] != x.shape[0]:
                raise ValueError((x.shape, y.shape))
            num_metrics = 1
        else:
            raise ValueError(y.shape)

        if self._config.tr_type != "morbo" and num_metrics != 1:
            raise ValueError(
                f"Single-objective mode requires num_metrics=1, got {num_metrics}"
            )

        if self._tr_state is None:
            self._tr_state = self._mode_impl.create_trust_region(
                self._num_dim, x.shape[0], self._rng, num_metrics=num_metrics
            )

        cfg_num_metrics = self._config.num_metrics
        if cfg_num_metrics is not None and num_metrics != cfg_num_metrics:
            raise ValueError(
                f"y has {num_metrics} metrics but expected {cfg_num_metrics}"
            )

        if self._expects_yvar is None:
            self._expects_yvar = y_var is not None
        if (y_var is not None) != bool(self._expects_yvar):
            raise ValueError(
                f"y_var must be {'provided' if self._expects_yvar else 'omitted'} on every tell() call"
            )
        if y_var is not None:
            y_var = np.asarray(y_var, dtype=float)
            if y_var.shape != y.shape:
                raise ValueError((y.shape, y_var.shape))
        if x.shape[0] == 0:
            if num_metrics == 1:
                return np.array([], dtype=float)
            return np.array([], dtype=float).reshape(0, num_metrics)

        x_unit = to_unit(x, self._bounds)
        self._x_obs_list.extend(x_unit.tolist())
        self._y_obs_list.extend(y.tolist())
        if y_var is not None:
            self._yvar_obs_list.extend(y_var.tolist())

        x_all = np.asarray(self._x_obs_list, dtype=float)
        y_all = np.asarray(self._y_obs_list, dtype=float)

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
        if prev_n > 0 and prev_n <= len(self._y_tr_list):
            if hasattr(self._tr_state, "best_value"):
                y_tr_array = np.asarray(self._y_tr_list, dtype=float)
                if y_tr_array.ndim == 2:
                    y_tr_array = y_tr_array[:, 0]
                self._tr_state.best_value = float(np.max(y_tr_array[:prev_n]))

        mu_all = np.asarray(self._y_tr_list, dtype=float)
        self._tr_state.update(mu_all)

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

    @staticmethod
    def _make_turbo_one_impl(config: TurboConfig) -> TurboModeImpl:
        from .turbo_one_impl import TurboOneImpl

        return TurboOneImpl(config)

    @staticmethod
    def _make_turbo_zero_impl(config: TurboConfig) -> TurboModeImpl:
        from .turbo_zero_impl import TurboZeroImpl

        return TurboZeroImpl(config)

    @staticmethod
    def _make_turbo_enn_impl(config: TurboConfig) -> TurboModeImpl:
        from .turbo_enn_impl import TurboENNImpl

        return TurboENNImpl(config)

    @staticmethod
    def _make_lhd_only_impl(config: TurboConfig) -> TurboModeImpl:
        from .lhd_only_impl import LHDOnlyImpl

        return LHDOnlyImpl(config)
