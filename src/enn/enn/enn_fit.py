from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

    from .enn import EpistemicNearestNeighbors
    from .enn_params import ENNParams


def subsample_loglik(
    model: EpistemicNearestNeighbors | Any,
    x: np.ndarray | Any,
    y: np.ndarray | Any,
    *,
    paramss: list[ENNParams] | list[Any],
    P: int = 10,
    rng: Generator | Any,
) -> list[float]:
    import numpy as np

    x_array = np.asarray(x, dtype=float)
    if x_array.ndim != 2:
        raise ValueError(x_array.shape)
    y_array = np.asarray(y, dtype=float)
    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)
    if y_array.ndim != 2:
        raise ValueError(y_array.shape)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError((x_array.shape, y_array.shape))
    if P <= 0:
        raise ValueError(P)
    if len(paramss) == 0:
        raise ValueError("paramss must be non-empty")
    n = x_array.shape[0]
    if n == 0:
        return [0.0] * len(paramss)
    if len(model) <= 1:
        return [0.0] * len(paramss)
    P_actual = min(P, n)
    if P_actual == n:
        indices = np.arange(n, dtype=int)
    else:
        indices = rng.permutation(n)[:P_actual]
    x_selected = x_array[indices]
    y_selected = y_array[indices]
    if not np.isfinite(y_selected).all():
        return [0.0] * len(paramss)
    post_batch = model.batch_posterior(
        x_selected, paramss, exclude_nearest=True, observation_noise=True
    )
    mu_batch = post_batch.mu
    se_batch = post_batch.se
    num_params = len(paramss)
    num_outputs = y_selected.shape[1]
    if mu_batch.shape != (num_params, P_actual, num_outputs) or se_batch.shape != (
        num_params,
        P_actual,
        num_outputs,
    ):
        raise ValueError(
            (
                mu_batch.shape,
                se_batch.shape,
                (num_params, P_actual, num_outputs),
            )
        )
    y_std = np.std(y_array, axis=0, keepdims=True).astype(float)
    y_std = np.where(np.isfinite(y_std) & (y_std > 0.0), y_std, 1.0)
    y_scaled = y_selected / y_std
    mu_scaled = mu_batch / y_std
    se_scaled = se_batch / y_std
    result = []
    for i in range(num_params):
        mu_i = mu_scaled[i]
        se_i = se_scaled[i]
        if not np.isfinite(mu_i).all() or not np.isfinite(se_i).all():
            result.append(0.0)
            continue
        if np.any(se_i <= 0.0):
            result.append(0.0)
            continue
        diff = y_scaled - mu_i
        var_scaled = se_i**2
        log_term = np.log(2.0 * np.pi * var_scaled)
        quad = diff**2 / var_scaled
        loglik = -0.5 * np.sum(log_term + quad)
        if not np.isfinite(loglik):
            result.append(0.0)
            continue
        result.append(float(loglik))
    return result


def enn_fit(
    model: EpistemicNearestNeighbors | Any,
    *,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int = 10,
    rng: Generator | Any,
    params_warm_start: ENNParams | Any | None = None,
) -> ENNParams:
    from .enn_params import ENNParams

    train_x = model.train_x
    train_y = model.train_y
    log_min = -3.0
    log_max = 3.0
    epi_var_scale_log_values = rng.uniform(log_min, log_max, size=num_fit_candidates)
    epi_var_scale_values = 10**epi_var_scale_log_values
    ale_homoscedastic_log_values = rng.uniform(
        log_min, log_max, size=num_fit_candidates
    )
    ale_homoscedastic_values = 10**ale_homoscedastic_log_values
    paramss = [
        ENNParams(
            k=k,
            epi_var_scale=float(epi_val),
            ale_homoscedastic_scale=float(ale_val),
        )
        for epi_val, ale_val in zip(epi_var_scale_values, ale_homoscedastic_values)
    ]
    if params_warm_start is not None:
        paramss.append(
            ENNParams(
                k=k,
                epi_var_scale=params_warm_start.epi_var_scale,
                ale_homoscedastic_scale=params_warm_start.ale_homoscedastic_scale,
            )
        )
    if len(paramss) == 0:
        return ENNParams(k=k, epi_var_scale=1.0, ale_homoscedastic_scale=0.0)
    import numpy as np

    logliks = subsample_loglik(
        model, train_x, train_y, paramss=paramss, P=num_fit_samples, rng=rng
    )
    if len(logliks) == 0:
        return paramss[0]
    best_idx = int(np.argmax(logliks))
    return paramss[best_idx]
