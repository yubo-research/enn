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

        if config is None:
            match mode:
                case TurboMode.TURBO_ONE:
                    config = TurboOneConfig()
                case TurboMode.TURBO_ZERO:
                    config = TurboZeroConfig()
                case TurboMode.TURBO_ENN:
                    config = TurboENNConfig()
                case TurboMode.LHD_ONLY:
                    config = LHDOnlyConfig()
                case _:
                    raise ValueError(f"Unknown mode: {mode}")
        else:
            match mode:
                case TurboMode.TURBO_ONE:
                    if not isinstance(config, TurboOneConfig):
                        raise ValueError(
                            f"mode={mode} requires TurboOneConfig, got {type(config).__name__}"
                        )
                case TurboMode.TURBO_ZERO:
                    if not isinstance(config, TurboZeroConfig):
                        raise ValueError(
                            f"mode={mode} requires TurboZeroConfig, got {type(config).__name__}"
                        )
                case TurboMode.TURBO_ENN:
                    if not isinstance(config, TurboENNConfig):
                        raise ValueError(
                            f"mode={mode} requires TurboENNConfig, got {type(config).__name__}"
                        )
                case TurboMode.LHD_ONLY:
                    if not isinstance(config, LHDOnlyConfig):
                        raise ValueError(
                            f"mode={mode} requires LHDOnlyConfig, got {type(config).__name__}"
                        )
                case _:
                    raise ValueError(f"Unknown mode: {mode}")
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
        self._y_tr_list: list[float] = []
        self._yvar_obs_list: list[float] | list[list[float]] = []
        self._expects_yvar: bool | None = None
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
        # For morbo, defer TR creation until tell() when we can infer num_metrics
        is_morbo = self._config.tr_type == "morbo"
        if self._tr_state is None and not is_morbo:
            self._tr_state = self._mode_impl.create_trust_region(
                self._num_dim, num_arms, self._rng
            )
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

        # For morbo, TR is created in tell() - if still None, return LHD
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

        # For morbo, TR is updated in tell() with raw multi-objective y
        if self._config.tr_type != "morbo":
            self._mode_impl.update_trust_region(
                self._tr_state,
                self._x_obs_list,
                self._y_tr_list,
                x_center=x_center,
                k=self._k,
            )
        return selected

    def _trim_trailing_obs(self) -> None:
        import numpy as np

        from .turbo_utils import argmax_random_tie

        if len(self._x_obs_list) <= self._trailing_obs:
            return
        y_tr_array = np.asarray(self._y_tr_list, dtype=float)
        incumbent_idx = argmax_random_tie(y_tr_array, rng=self._rng)
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
        incumbent_value = y_tr_array[incumbent_idx]
        self._x_obs_list = x_array[indices].tolist()
        y_obs_array = np.asarray(self._y_obs_list, dtype=float)
        self._y_obs_list = y_obs_array[indices].tolist()
        self._y_tr_list = y_tr_array[indices].tolist()
        if len(self._yvar_obs_list) == len(y_obs_array):
            yvar_array = np.asarray(self._yvar_obs_list, dtype=float)
            self._yvar_obs_list = yvar_array[indices].tolist()
        y_trimmed = np.asarray(self._y_tr_list, dtype=float)
        if not np.any(np.abs(y_trimmed - incumbent_value) < 1e-10):
            raise RuntimeError("Incumbent value must be preserved in trimmed list")

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

        # morbo accepts 2D y with shape (n, num_metrics)
        is_morbo = self._config.tr_type == "morbo"
        if is_morbo:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.ndim != 2 or y.shape[0] != x.shape[0]:
                raise ValueError((x.shape, y.shape))
            num_metrics = y.shape[1]
            # Create TR lazily for morbo, inferring num_metrics from y
            if self._tr_state is None:
                self._tr_state = self._mode_impl.create_trust_region(
                    self._num_dim, x.shape[0], self._rng, num_metrics=num_metrics
                )
            cfg_num_metrics = self._config.num_metrics
            if cfg_num_metrics is not None and num_metrics != cfg_num_metrics:
                raise ValueError(
                    f"y has {num_metrics} metrics but expected {cfg_num_metrics}"
                )
        else:
            if self._tr_state is None:
                raise ValueError("tell() called before ask()")
            if y.ndim != 1 or y.shape[0] != x.shape[0]:
                raise ValueError((x.shape, y.shape))

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
            return np.array([], dtype=float)
        x_unit = to_unit(x, self._bounds)
        self._x_obs_list.extend(x_unit.tolist())

        if is_morbo:
            y_estimate = y
            self._y_obs_list.extend(y.tolist())
            if y_var is not None:
                self._yvar_obs_list.extend(y_var.tolist())
            y_all = np.asarray(self._y_obs_list, dtype=float)
            if y_all.ndim == 1:
                y_all = y_all.reshape(-1, num_metrics)
            x_all = np.asarray(self._x_obs_list, dtype=float)
            self._tr_state.update_xy(x_all, y_all, k=self._k)
        else:
            from .turbo_mode import TurboMode

            self._y_obs_list.extend(y.tolist())
            if y_var is not None:
                self._yvar_obs_list.extend(y_var.tolist())

            if self._mode in (TurboMode.TURBO_ONE, TurboMode.TURBO_ENN):
                self._mode_impl.prepare_ask(
                    self._x_obs_list,
                    self._y_obs_list,
                    self._yvar_obs_list,
                    self._num_dim,
                    0,
                    rng=self._rng,
                )
                x_all = np.asarray(self._x_obs_list, dtype=float)
                y_all = np.asarray(self._y_obs_list, dtype=float)
                if self._mode == TurboMode.TURBO_ONE:
                    # We intentionally evaluate the GP posterior at the training inputs
                    # (the observed points) right after conditioning the model. GPyTorch
                    # warns about this in debug mode, but it's expected for our TR logic.
                    import warnings

                    try:
                        from gpytorch.utils.warnings import GPInputWarning
                    except Exception:  # pragma: no cover
                        GPInputWarning = None

                    if GPInputWarning is None:
                        mu_all = np.asarray(
                            self._mode_impl.estimate_y(x_all, y_all), dtype=float
                        ).reshape(-1)
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message=r"The input matches the stored training data\..*",
                                category=GPInputWarning,
                            )
                            mu_all = np.asarray(
                                self._mode_impl.estimate_y(x_all, y_all), dtype=float
                            ).reshape(-1)
                else:
                    mu_all = np.asarray(
                        self._mode_impl.estimate_y(x_all, y_all), dtype=float
                    ).reshape(-1)
                self._y_tr_list = mu_all.tolist()
                if self._mode == TurboMode.TURBO_ONE:
                    import warnings

                    try:
                        from gpytorch.utils.warnings import GPInputWarning
                    except Exception:  # pragma: no cover
                        GPInputWarning = None

                    if GPInputWarning is None:
                        y_estimate = np.asarray(
                            self._mode_impl.estimate_y(x_unit, y), dtype=float
                        )
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message=r"The input matches the stored training data\..*",
                                category=GPInputWarning,
                            )
                            y_estimate = np.asarray(
                                self._mode_impl.estimate_y(x_unit, y), dtype=float
                            )
                else:
                    y_estimate = np.asarray(
                        self._mode_impl.estimate_y(x_unit, y), dtype=float
                    )
            else:
                y_estimate = self._mode_impl.estimate_y(x_unit, y)
                self._y_tr_list.extend(np.asarray(y_estimate, dtype=float).tolist())

            if self._trailing_obs is not None:
                self._trim_trailing_obs()
            prev_n = int(getattr(self._tr_state, "prev_num_obs", 0))
            if prev_n > 0 and prev_n <= len(self._y_tr_list):
                if hasattr(self._tr_state, "best_value"):
                    self._tr_state.best_value = float(
                        np.max(np.asarray(self._y_tr_list, dtype=float)[:prev_n])
                    )
            self._mode_impl.update_trust_region(
                self._tr_state, self._x_obs_list, self._y_tr_list, k=self._k
            )

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
