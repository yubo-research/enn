from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from . import turbo_optimizer_utils, turbo_utils
from .components import AcquisitionOptimizer, Surrogate
from .config.enums import CandidateRV
from .strategies import OptimizationStrategy
from .types.appendable_array import AppendableArray

if TYPE_CHECKING:
    from numpy.random import Generator

    from .config.optimizer_config import OptimizerConfig


class Optimizer:
    def __init__(
        self,
        *,
        bounds: np.ndarray,
        config: OptimizerConfig,
        rng: Generator,
        surrogate: Surrogate,
        acquisition_optimizer: AcquisitionOptimizer,
        strategy: OptimizationStrategy | None = None,
    ) -> None:
        self._config = config
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must be (d, 2), got {bounds.shape}")
        self._bounds = bounds
        self._num_dim = bounds.shape[0]
        self._rng = rng
        self._surrogate = surrogate
        self._acq_optimizer = acquisition_optimizer
        self._strategy = (
            strategy
            if strategy is not None
            else config.init.init_strategy.create_runtime_strategy(
                bounds=self._bounds, rng=self._rng, num_init=config.init.num_init
            )
        )
        self._tr_state = config.trust_region.build(
            num_dim=self._num_dim,
            rng=rng,
            candidate_rv=config.candidate_rv,
        )
        self._trailing_obs = (
            None if config.trailing_obs is None else int(config.trailing_obs)
        )
        self._gp_num_steps = 50
        if self._trailing_obs is not None and self._trailing_obs <= 0:
            raise ValueError(f"trailing_obs must be > 0, got {self._trailing_obs}")
        self._x_obs = AppendableArray()
        self._y_obs = AppendableArray()
        self._yvar_obs = AppendableArray()
        self._y_tr_list: list[float] | list[list[float]] = []
        self._expects_yvar: bool | None = None
        self._dt_fit = 0.0
        self._dt_gen = 0.0
        self._dt_sel = 0.0
        self._dt_tell = 0.0
        self._sobol_seed_base = int(rng.integers(2**31 - 1))
        self._restart_generation = 0
        self._incumbent_idx: int | None = None
        self._incumbent_x_unit: np.ndarray | None = None
        self._incumbent_y_scalar: np.ndarray | None = None

    @property
    def tr_obs_count(self) -> int:
        return len(self._y_obs)

    @property
    def tr_length(self) -> float:
        return float(self._tr_state.length)

    def telemetry(self) -> turbo_utils.Telemetry:
        return turbo_utils.Telemetry(
            dt_fit=self._dt_fit,
            dt_gen=self._dt_gen,
            dt_sel=self._dt_sel,
            dt_tell=self._dt_tell,
        )

    @property
    def init_progress(self) -> tuple[int, int] | None:
        return self._strategy.init_progress()

    def ask(self, num_arms: int) -> np.ndarray:
        num_arms = int(num_arms)
        if num_arms <= 0:
            raise ValueError(num_arms)
        turbo_optimizer_utils.reset_timing(self)
        return self._strategy.ask(self, num_arms)

    def _ask_normal(self, num_arms: int, *, is_fallback: bool = False) -> np.ndarray:
        self._tr_state.validate_request(num_arms, is_fallback=is_fallback)
        self._maybe_resample_weights()
        x_center = self._incumbent_x_unit
        if x_center is None:
            if len(self._y_obs) == 0:
                raise RuntimeError("no observations")
            x_center = np.full(self._num_dim, 0.5)
        t0 = time.perf_counter()
        lengthscales = self._surrogate.lengthscales
        x_cand = self._generate_candidates(x_center, lengthscales, num_arms=num_arms)
        self._dt_gen = time.perf_counter() - t0
        t0 = time.perf_counter()
        selected = self._acq_optimizer.select(
            x_cand,
            num_arms,
            self._surrogate,
            self._rng,
            tr_state=self._tr_state,
        )
        self._dt_sel = time.perf_counter() - t0
        return turbo_utils.from_unit(selected, self._bounds)

    def _find_x_center(self, x_obs: np.ndarray, y_obs: np.ndarray) -> np.ndarray | None:
        return self._incumbent_x_unit

    def _maybe_resample_weights(self) -> None:
        from .config.rescalarize import Rescalarize

        if hasattr(self._tr_state, "rescalarize"):
            if self._tr_state.rescalarize == Rescalarize.ON_PROPOSE:
                self._tr_state.resample_weights(self._rng)

    def _generate_candidates(
        self,
        x_center: np.ndarray,
        lengthscales: np.ndarray | None,
        *,
        num_arms: int,
    ) -> np.ndarray:
        from . import tr_helpers

        if lengthscales is not None:
            lengthscales = np.asarray(lengthscales, dtype=float).reshape(-1)
            if not np.all(np.isfinite(lengthscales)):
                raise ValueError("lengthscales must be finite")
        num_candidates = int(
            self._config.candidates.num_candidates(
                num_dim=self._num_dim, num_arms=num_arms
            )
        )
        if num_candidates <= 0:
            raise ValueError(num_candidates)
        candidate_rv = self._config.candidate_rv
        if candidate_rv == CandidateRV.SOBOL:
            from scipy.stats import qmc

            sobol_seed = turbo_optimizer_utils.sobol_seed_for_state(
                self._sobol_seed_base,
                restart_generation=self._restart_generation,
                n_obs=len(self._x_obs),
                num_arms=num_arms,
            )
            sobol_engine = qmc.Sobol(d=self._num_dim, scramble=True, seed=sobol_seed)
        else:
            sobol_engine = None
        if getattr(self._tr_state, "uses_custom_candidate_gen", False):
            return self._tr_state.generate_candidates(
                x_center,
                lengthscales,
                num_candidates,
                rng=self._rng,
                sobol_engine=sobol_engine,
                raasp_driver=self._config.raasp_driver,
                num_pert=20,
            )
        return tr_helpers.generate_tr_candidates(
            self._tr_state.compute_bounds_1d,
            x_center,
            lengthscales,
            num_candidates,
            rng=self._rng,
            candidate_rv=candidate_rv,
            sobol_engine=sobol_engine,
            raasp_driver=self._config.raasp_driver,
            num_pert=20,
        )

    def _validate_tell_inputs(
        self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None
    ) -> turbo_optimizer_utils.TellInputs:
        inputs = turbo_optimizer_utils.validate_tell_inputs(x, y, y_var, self._num_dim)
        tr_num_metrics = getattr(self._tr_state, "num_metrics", 1)
        if inputs.num_metrics != tr_num_metrics:
            raise ValueError(
                f"y has {inputs.num_metrics} metrics but trust region expects {tr_num_metrics}"
            )
        if self._expects_yvar is None:
            self._expects_yvar = inputs.y_var is not None
        if (inputs.y_var is not None) != bool(self._expects_yvar):
            raise ValueError(
                f"y_var must be {'provided' if self._expects_yvar else 'omitted'} on every tell()"
            )
        return inputs

    def _update_incumbent(self) -> None:
        if len(self._y_obs) == 0:
            self._incumbent_idx = None
            self._incumbent_x_unit = None
            self._incumbent_y_scalar = None
            return
        x_obs = self._x_obs.view()
        y_obs = self._y_obs.view()
        candidate_indices = self._surrogate.get_incumbent_candidate_indices(y_obs)
        x_cand = x_obs[candidate_indices]
        y_cand = y_obs[candidate_indices]
        mu_cand = None
        noise_aware = False
        if hasattr(self._tr_state, "incumbent_selector"):
            noise_aware = getattr(
                self._tr_state.incumbent_selector, "noise_aware", False
            )
        elif hasattr(self._tr_state, "config"):
            noise_aware = getattr(self._tr_state.config, "noise_aware", False)
        if noise_aware:
            try:
                mu_cand = self._surrogate.predict(x_cand).mu
            except RuntimeError:
                mu_cand = None
        idx_in_cand = self._tr_state.get_incumbent_index(y_cand, self._rng, mu=mu_cand)
        self._incumbent_idx = int(candidate_indices[idx_in_cand])
        self._incumbent_x_unit = x_obs[self._incumbent_idx]
        if noise_aware and mu_cand is not None:
            self._incumbent_y_scalar = mu_cand[idx_in_cand : idx_in_cand + 1].copy()
        else:
            self._incumbent_y_scalar = y_cand[idx_in_cand : idx_in_cand + 1].copy()

    def _trim_trailing_obs(self) -> None:
        incumbent_indices = np.array([self._incumbent_idx], dtype=int)
        obs = turbo_optimizer_utils.trim_trailing_observations(
            self._x_obs.view().tolist(),
            self._y_obs.view().tolist(),
            self._y_tr_list,
            self._yvar_obs.view().tolist() if len(self._yvar_obs) > 0 else [],
            trailing_obs=self._trailing_obs,
            incumbent_indices=incumbent_indices,
        )
        self._x_obs = AppendableArray()
        for x in obs.x_obs:
            self._x_obs.append(np.array(x))
        self._y_obs = AppendableArray()
        for y in obs.y_obs:
            self._y_obs.append(np.array(y))
        self._yvar_obs = AppendableArray()
        if obs.yvar_obs:
            for yvar in obs.yvar_obs:
                self._yvar_obs.append(np.array(yvar))
        self._y_tr_list = obs.y_tr

    def _update_best_value_if_needed(self) -> None:
        pass

    def tell(
        self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None
    ) -> np.ndarray:
        with turbo_utils.record_duration(
            lambda dt: setattr(self, "_dt_tell", float(dt))
        ):
            inputs = self._validate_tell_inputs(x, y, y_var)
            if inputs.x.shape[0] == 0:
                return (
                    np.array([], dtype=float)
                    if inputs.num_metrics == 1
                    else np.empty((0, inputs.num_metrics), dtype=float)
                )
            x_unit = turbo_utils.to_unit(inputs.x, self._bounds)
            for i in range(inputs.x.shape[0]):
                self._x_obs.append(x_unit[i])
                self._y_obs.append(inputs.y[i])
                if inputs.y_var is not None:
                    self._yvar_obs.append(inputs.y_var[i])
            return self._strategy.tell(self, inputs, x_unit=x_unit)


def create_optimizer(
    *,
    bounds: np.ndarray,
    config: OptimizerConfig,
    rng: Generator,
) -> Optimizer:
    from .components.builder import build_acquisition_optimizer, build_surrogate

    surrogate = build_surrogate(config)
    acq_optimizer = build_acquisition_optimizer(config)
    return Optimizer(
        bounds=bounds,
        config=config,
        rng=rng,
        surrogate=surrogate,
        acquisition_optimizer=acq_optimizer,
    )
