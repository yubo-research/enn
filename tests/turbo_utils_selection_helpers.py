"""GP / uniform candidate selection helpers used only from turbo utils tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.random import Generator

if TYPE_CHECKING:
    from enn.turbo.turbo_gp import TurboGP


def select_uniform(
    x_cand: np.ndarray,
    num_arms: int,
    num_dim: int,
    rng: Generator,
    from_unit_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    if x_cand.ndim != 2 or x_cand.shape[1] != num_dim:
        raise ValueError(x_cand.shape)
    if x_cand.shape[0] < num_arms:
        raise ValueError((x_cand.shape[0], num_arms))
    idx = rng.choice(x_cand.shape[0], size=num_arms, replace=False)
    return from_unit_fn(x_cand[idx])


def select_gp_thompson(
    x_cand: np.ndarray,
    num_arms: int,
    x_obs_list: list,
    y_obs_list: list,
    num_dim: int,
    *,
    gp_num_steps: int,
    rng: Generator,
    gp_y_stats: tuple[float, float],
    select_sobol_fn: Callable[[np.ndarray, int], np.ndarray],
    from_unit_fn: Callable[[np.ndarray], np.ndarray],
    model: TurboGP | None = None,
) -> tuple[np.ndarray, tuple[float, float], TurboGP | None]:
    from enn.turbo.turbo_gp_fit import fit_gp
    from enn.turbo.turbo_utils import gp_thompson_sample

    gp_y_mean, gp_y_std = gp_y_stats
    if len(x_obs_list) == 0:
        return select_sobol_fn(x_cand, num_arms), (gp_y_mean, gp_y_std), None
    fitted_mean, fitted_std = gp_y_mean, gp_y_std
    if model is None:
        gp_result = fit_gp(
            x_obs_list,
            y_obs_list,
            num_dim,
            num_steps=gp_num_steps,
        )
        model, fitted_mean, fitted_std = (
            gp_result.model,
            gp_result.y_mean,
            gp_result.y_std,
        )
    if model is None:
        return select_sobol_fn(x_cand, num_arms), (gp_y_mean, gp_y_std), None
    if x_cand.shape[0] < num_arms:
        raise ValueError((x_cand.shape[0], num_arms))
    idx = gp_thompson_sample(
        model, x_cand, num_arms, rng, gp_y_mean=fitted_mean, gp_y_std=fitted_std
    )
    return from_unit_fn(x_cand[idx]), (fitted_mean, fitted_std), model
