from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams
    from .turbo_gp import TurboGP
    from .config.enums import ENNIndexDriver


def mk_enn(
    x_obs_list: list[float] | list[list[float]],
    y_obs_list: list[float] | list[list[float]],
    k: int,
    yvar_obs_list: list[float] | None = None,
    *,
    num_fit_samples: int | None = None,
    num_fit_candidates: int | None = None,
    scale_x: bool = False,
    index_driver: ENNIndexDriver | Any | None = None,
    rng: Generator | Any | None = None,
    params_warm_start: ENNParams | Any | None = None,
) -> tuple[EpistemicNearestNeighbors | None, ENNParams | None]:
    import numpy as np
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams
    from .config.enums import ENNIndexDriver

    if index_driver is None:
        index_driver = ENNIndexDriver.FLAT

    if len(x_obs_list) == 0:
        return None, None
    y_obs_array = np.asarray(y_obs_list, dtype=float)
    if y_obs_array.size == 0:
        return None, None
    if y_obs_array.ndim == 1:
        y = y_obs_array.reshape(-1, 1)
    else:
        y = y_obs_array
    if yvar_obs_list is not None and len(yvar_obs_list) > 0:
        yvar_array = np.asarray(yvar_obs_list, dtype=float)
        if yvar_array.ndim == 1:
            yvar = yvar_array.reshape(-1, 1)
        else:
            yvar = yvar_array
    else:
        yvar = None
    x_obs_array = np.asarray(x_obs_list, dtype=float)
    enn_model = EpistemicNearestNeighbors(
        x_obs_array,
        y,
        yvar,
        scale_x=scale_x,
        index_driver=index_driver,
    )
    if len(enn_model) == 0:
        return None, None
    fitted_params: ENNParams | None = None
    if num_fit_samples is not None and rng is not None:
        from enn.enn.enn_fit import enn_fit

        fitted_params = enn_fit(
            enn_model,
            k=k,
            num_fit_candidates=num_fit_candidates
            if num_fit_candidates is not None
            else 30,
            num_fit_samples=num_fit_samples,
            rng=rng,
            params_warm_start=params_warm_start,
        )
    else:
        fitted_params = ENNParams(
            k_num_neighbors=k,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.0,
        )
    return enn_model, fitted_params


def select_uniform(
    x_cand: np.ndarray,
    num_arms: int,
    num_dim: int,
    rng: Generator | Any,
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
    rng: Generator | Any,
    gp_y_stats: tuple[float, float],
    select_sobol_fn: Callable[[np.ndarray, int], np.ndarray],
    from_unit_fn: Callable[[np.ndarray], np.ndarray],
    model: TurboGP | None = None,
) -> tuple[np.ndarray, tuple[float, float], TurboGP | None]:
    from .turbo_gp_fit import fit_gp

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
    from .turbo_utils import gp_thompson_sample

    idx = gp_thompson_sample(
        model, x_cand, num_arms, rng, gp_y_mean=fitted_mean, gp_y_std=fitted_std
    )
    return from_unit_fn(x_cand[idx]), (fitted_mean, fitted_std), model
