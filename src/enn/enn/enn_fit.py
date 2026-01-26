from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from .enn_class import EpistemicNearestNeighbors
    from .enn_params import ENNParams


def _validate_subsample_inputs(
    x: np.ndarray, y: np.ndarray, P: int, paramss: list
) -> tuple[np.ndarray, np.ndarray]:
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
    return x_array, y_array


def _compute_single_loglik(
    y_scaled: np.ndarray, mu_i: np.ndarray, se_i: np.ndarray
) -> float:
    if not np.isfinite(mu_i).all() or not np.isfinite(se_i).all():
        return 0.0
    if np.any(se_i <= 0.0):
        return 0.0
    var_scaled = se_i**2
    loglik = -0.5 * np.sum(
        np.log(2.0 * np.pi * var_scaled) + (y_scaled - mu_i) ** 2 / var_scaled
    )
    return float(loglik) if np.isfinite(loglik) else 0.0


def subsample_loglik(
    model: EpistemicNearestNeighbors,
    x: np.ndarray,
    y: np.ndarray,
    *,
    paramss: list[ENNParams],
    P: int = 10,
    rng: Generator,
    y_std: np.ndarray | None = None,
) -> list[float]:
    x_array, y_array = _validate_subsample_inputs(x, y, P, paramss)
    n = x_array.shape[0]
    if n == 0 or len(model) <= 1:
        return [0.0] * len(paramss)
    P_actual = min(P, n)
    indices = (
        np.arange(n, dtype=int) if P_actual == n else rng.permutation(n)[:P_actual]
    )
    x_sel, y_sel = x_array[indices], y_array[indices]
    if not np.isfinite(y_sel).all():
        return [0.0] * len(paramss)
    from .enn_params import PosteriorFlags

    post = model.batch_posterior(
        x_sel,
        paramss,
        flags=PosteriorFlags(exclude_nearest=True, observation_noise=True),
    )
    num_params, num_outputs = len(paramss), y_sel.shape[1]
    expected_shape = (num_params, P_actual, num_outputs)
    if post.mu.shape != expected_shape or post.se.shape != expected_shape:
        raise ValueError((post.mu.shape, post.se.shape, expected_shape))
    if y_std is None:
        y_std = np.std(y_array, axis=0, keepdims=True).astype(float)
    y_std = np.where(np.isfinite(y_std) & (y_std > 0.0), y_std, 1.0)
    y_scaled = y_sel / y_std
    mu_scaled = post.mu / y_std
    se_scaled = post.se / y_std
    return [
        _compute_single_loglik(y_scaled, mu_scaled[i], se_scaled[i])
        for i in range(num_params)
    ]


def enn_fit(
    model: EpistemicNearestNeighbors,
    *,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int = 10,
    rng: Generator,
    params_warm_start: ENNParams | None = None,
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
            k_num_neighbors=k,
            epistemic_variance_scale=float(epi_val),
            aleatoric_variance_scale=float(ale_val),
        )
        for epi_val, ale_val in zip(epi_var_scale_values, ale_homoscedastic_values)
    ]
    if params_warm_start is not None:
        paramss.append(
            ENNParams(
                k_num_neighbors=k,
                epistemic_variance_scale=params_warm_start.epistemic_variance_scale,
                aleatoric_variance_scale=params_warm_start.aleatoric_variance_scale,
            )
        )
    if len(paramss) == 0:
        return ENNParams(
            k_num_neighbors=k,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.0,
        )

    y_std = np.std(train_y, axis=0, keepdims=True).astype(float)
    logliks = subsample_loglik(
        model,
        train_x,
        train_y,
        paramss=paramss,
        P=num_fit_samples,
        rng=rng,
        y_std=y_std,
    )
    if len(logliks) == 0:
        return paramss[0]
    best_idx = int(np.argmax(logliks))
    return paramss[best_idx]
