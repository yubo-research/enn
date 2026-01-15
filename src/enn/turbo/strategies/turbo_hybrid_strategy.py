from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..sampling import draw_lhd
from .optimization_strategy import OptimizationStrategy

if TYPE_CHECKING:
    from ..optimizer import Optimizer
    from ..types import TellInputs


@dataclass
class TurboHybridStrategy(OptimizationStrategy):
    _bounds: np.ndarray
    _num_dim: int
    _rng: object
    _num_init: int

    _init_lhd: np.ndarray
    _init_idx: int = 0

    @classmethod
    def create(
        cls, *, bounds: np.ndarray, rng: object, num_init: int | None
    ) -> TurboHybridStrategy:
        from numpy.random import Generator

        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must be (d, 2), got {bounds.shape}")
        num_dim = int(bounds.shape[0])

        if not isinstance(rng, Generator):
            raise TypeError("rng must be a numpy.random.Generator")

        n_init = int(num_init if num_init is not None else 2 * num_dim)
        if n_init <= 0:
            raise ValueError(f"num_init must be > 0, got {n_init}")

        init_lhd = draw_lhd(bounds=bounds, num_arms=n_init, rng=rng)
        return cls(
            _bounds=bounds,
            _num_dim=num_dim,
            _rng=rng,
            _num_init=n_init,
            _init_lhd=init_lhd,
        )

    def _reset_init(self) -> None:
        from numpy.random import Generator

        if not isinstance(self._rng, Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self._init_lhd = draw_lhd(
            bounds=self._bounds, num_arms=self._num_init, rng=self._rng
        )
        self._init_idx = 0

    def _get_init_points(self, num_arms: int, *, fallback_fn=None) -> np.ndarray:
        num_arms = int(num_arms)
        num_to_return = min(num_arms, self._num_init - self._init_idx)
        result = self._init_lhd[self._init_idx : self._init_idx + num_to_return]
        self._init_idx += num_to_return

        if num_to_return < num_arms:
            extra = (
                fallback_fn
                or (lambda n: draw_lhd(bounds=self._bounds, num_arms=n, rng=self._rng))
            )(num_arms - num_to_return)
            result = np.vstack([result, extra])

        return result

    def ask(self, opt: Optimizer, num_arms: int) -> np.ndarray:
        if opt._tr_state.needs_restart():
            opt._tr_state.restart(opt._rng)
            opt._restart_generation += 1
            opt._x_obs_list.clear()
            opt._y_obs_list.clear()
            opt._yvar_obs_list.clear()
            opt._y_tr_list = []
            self._reset_init()
            return self._get_init_points(num_arms)

        if self._init_idx < self._num_init:

            def fallback(n: int) -> np.ndarray:
                return opt._ask_normal(n, is_fallback=True)

            return self._get_init_points(
                num_arms,
                fallback_fn=fallback if opt._x_obs_list else None,
            )

        if not opt._x_obs_list:
            return draw_lhd(bounds=self._bounds, num_arms=num_arms, rng=self._rng)

        opt._tr_state.validate_request(int(num_arms))

        return opt._ask_normal(int(num_arms))

    def init_progress(self) -> tuple[int, int] | None:
        return (int(self._init_idx), int(self._num_init))

    def tell(
        self, opt: Optimizer, inputs: TellInputs, *, x_unit: np.ndarray
    ) -> np.ndarray:
        x_all = np.asarray(opt._x_obs_list, dtype=float)
        y_all = np.asarray(opt._y_obs_list, dtype=float)
        y_var_all = (
            np.asarray(opt._yvar_obs_list, dtype=float) if opt._yvar_obs_list else None
        )

        opt._surrogate.fit(x_all, y_all, y_var_all, num_steps=0, rng=opt._rng)

        try:
            posterior = opt._surrogate.predict(x_all)
            y_tr = posterior.mu
        except RuntimeError:
            y_tr = y_all
        if y_tr.ndim == 2 and y_tr.shape[1] == 1:
            opt._y_tr_list = y_tr[:, 0].tolist()
        else:
            opt._y_tr_list = y_tr.tolist()

        try:
            new_posterior = opt._surrogate.predict(x_unit)
            y_estimate = np.asarray(new_posterior.mu, dtype=float)
        except RuntimeError:
            y_estimate = np.asarray(inputs.y, dtype=float)

        if opt._trailing_obs is not None:
            opt._trim_trailing_obs()

        opt._update_best_value_if_needed()
        opt._tr_state.update(np.asarray(opt._y_tr_list, dtype=float))

        return y_estimate
