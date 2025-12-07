from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn_params import ENNParams

    from .turbo_gp import TurboGP

from enn.enn_util import standardize_y

from .turbo_utils import gp_thompson_sample


def mk_enn(
    x_obs_list: list[float] | list[list[float]],
    y_obs_list: list[float] | list[list[float]],
    *,
    yvar_obs_list: list[float] | None = None,
    k: int,
    num_fit_samples: int | None = None,
    rng: Generator | Any | None = None,
) -> tuple[EpistemicNearestNeighbors | None, ENNParams | None, float, float]:
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn_params import ENNParams

    if len(x_obs_list) == 0:
        return None, None, 0.0, 1.0
    y_obs_array = np.asarray(y_obs_list, dtype=float)
    if y_obs_array.size == 0:
        return None, None, 0.0, 1.0

    mu_y, sigma_y = standardize_y(y_obs_array)
    y_standardized = (y_obs_array - mu_y) / sigma_y

    y = y_standardized.reshape(-1, 1)
    if yvar_obs_list is not None and len(yvar_obs_list) > 0:
        yvar_array = np.asarray(yvar_obs_list, dtype=float)
        # Scale variance by sigma_y^2 to match standardized y
        yvar = (yvar_array / (sigma_y**2)).reshape(-1, 1)
    else:
        yvar = np.zeros_like(y, dtype=float)
    x_obs_array = np.asarray(x_obs_list, dtype=float)
    enn_model = EpistemicNearestNeighbors(
        x_obs_array,
        y,
        yvar,
    )
    if len(enn_model) == 0:
        return None, None, 0.0, 1.0

    fitted_params: ENNParams | None = None
    if num_fit_samples is not None and rng is not None:
        from enn.enn_fit import enn_fit

        fitted_params = enn_fit(
            enn_model,
            k=k,
            num_fit_candidates=30,
            num_fit_samples=num_fit_samples,
            rng=rng,
        )
    else:
        fitted_params = ENNParams(k=k, var_scale=1.0)

    return enn_model, fitted_params, mu_y, sigma_y


def select_enn_pareto(
    x_cand: np.ndarray,
    num_arms: int,
    x_obs_list: list[float] | list[list[float]],
    y_obs_list: list[float] | list[list[float]],
    k: int | None,
    var_scale: float,
    rng: Generator | Any,
    fallback_fn: Callable[[np.ndarray, int], np.ndarray],
    from_unit_fn: Callable[[np.ndarray], np.ndarray],
    *,
    enn_model: EpistemicNearestNeighbors | None = None,
    fitted_params: ENNParams | None = None,
) -> np.ndarray:
    from enn.enn_params import ENNParams
    from enn.enn_util import arms_from_pareto_fronts

    if enn_model is None:
        if k is None:
            k = 10
        enn_model, _, _, _ = mk_enn(
            x_obs_list,
            y_obs_list,
            k=k,
        )
    if enn_model is None:
        return fallback_fn(x_cand, num_arms)

    if fitted_params is not None:
        params = fitted_params
    else:
        if k is None:
            k = 10
        params = ENNParams(k=k, var_scale=var_scale)

    posterior = enn_model.posterior(x_cand, params=params)
    mu = posterior.mu[:, 0]
    se = posterior.se[:, 0]

    x_arms = arms_from_pareto_fronts(x_cand, mu, se, num_arms, rng)
    return from_unit_fn(x_arms)


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
    x_obs_list: list[float] | list[list[float]],
    y_obs_list: list[float] | list[list[float]],
    num_dim: int,
    gp_num_steps: int,
    rng: Generator | Any,
    gp_y_mean: float,
    gp_y_std: float,
    select_sobol_fn: Callable[[np.ndarray, int], np.ndarray],
    from_unit_fn: Callable[[np.ndarray], np.ndarray],
    *,
    model: TurboGP | None = None,
    new_gp_y_mean: float | None = None,
    new_gp_y_std: float | None = None,
) -> tuple[np.ndarray, float, float, TurboGP | None]:
    from .turbo_utils import fit_gp

    if len(x_obs_list) == 0:
        return select_sobol_fn(x_cand, num_arms), gp_y_mean, gp_y_std, None
    if model is None:
        model, _likelihood, new_gp_y_mean, new_gp_y_std = fit_gp(
            x_obs_list,
            y_obs_list,
            num_dim,
            num_steps=gp_num_steps,
        )
    if model is None:
        return select_sobol_fn(x_cand, num_arms), gp_y_mean, gp_y_std, None
    if new_gp_y_mean is None:
        new_gp_y_mean = gp_y_mean
    if new_gp_y_std is None:
        new_gp_y_std = gp_y_std
    if x_cand.shape[0] < num_arms:
        raise ValueError((x_cand.shape[0], num_arms))
    idx = gp_thompson_sample(
        model,
        x_cand,
        num_arms,
        rng,
        new_gp_y_mean,
        new_gp_y_std,
    )
    return from_unit_fn(x_cand[idx]), new_gp_y_mean, new_gp_y_std, model
