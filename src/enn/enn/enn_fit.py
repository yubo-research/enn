from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from .enn_class import EpistemicNearestNeighbors
    from .enn_params import ENNParams

try:
    from enn._rust import subsample_loglik as _rust_subsample_loglik
    from enn._rust import enn_fit as _rust_enn_fit
    from enn._rust import ENNParams as RustENNParams
except ImportError:  # pragma: no cover
    _rust_subsample_loglik = None
    _rust_enn_fit = None
    RustENNParams = None


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


def _subsample_loglik_python(
    model: EpistemicNearestNeighbors,
    x: np.ndarray,
    y: np.ndarray,
    *,
    paramss: list[ENNParams],
    P: int = 10,
    rng: Generator,
    y_std: np.ndarray | None = None,
) -> list[float]:
    """Python fallback for subsample log-likelihood."""
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
    """Compute subsample log-likelihood using Rust backend if available, otherwise Python."""
    from .enn_class import EpistemicNearestNeighbors as PyENN

    # Validate inputs
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)

    if not isinstance(model, PyENN):
        raise TypeError(f"Expected EpistemicNearestNeighbors, got {type(model)}")

    # Use Python fallback if Rust is not available or model doesn't have Rust backend
    if _rust_subsample_loglik is None or model.rust_backend is None:
        return _subsample_loglik_python(
            model, x, y, paramss=paramss, P=P, rng=rng, y_std=y_std
        )

    # Convert Python RNG to seed for Rust RNG
    seed = int(rng.integers(0, 2**63 - 1))

    # Convert Python ENNParams to lists for Rust
    k_values = [p.k_num_neighbors for p in paramss]
    epi_scales = [p.epistemic_variance_scale for p in paramss]
    ale_scales = [p.aleatoric_variance_scale for p in paramss]

    # Convert y_std if provided
    y_std_arr = None
    if y_std is not None:
        y_std_arr = np.asarray(y_std, dtype=float).ravel()

    return _rust_subsample_loglik(
        model.rust_backend,
        x_array,
        y_array,
        k_values,
        epi_scales,
        ale_scales,
        P,
        seed,
        y_std_arr,
    )


def _enn_fit_python(
    model: EpistemicNearestNeighbors,
    *,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int = 10,
    rng: Generator,
    params_warm_start: ENNParams | None = None,
    infer_aleatoric_variance_scale: bool = True,
) -> ENNParams:
    """Python fallback for enn_fit."""
    from .enn_params import ENNParams

    train_x = model.train_x
    train_y = model.train_y
    log_min = -3.0
    log_max = 3.0
    epi_var_scale_log_values = rng.uniform(log_min, log_max, size=num_fit_candidates)
    epi_var_scale_values = 10**epi_var_scale_log_values
    ale_homoscedastic_values = (
        10 ** rng.uniform(log_min, log_max, size=num_fit_candidates)
        if infer_aleatoric_variance_scale
        else np.zeros(num_fit_candidates, dtype=float)
    )
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
                aleatoric_variance_scale=(
                    params_warm_start.aleatoric_variance_scale
                    if infer_aleatoric_variance_scale
                    else 0.0
                ),
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


def enn_fit(
    model: EpistemicNearestNeighbors,
    *,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int = 10,
    rng: Generator,
    params_warm_start: ENNParams | None = None,
    infer_aleatoric_variance_scale: bool = True,
) -> ENNParams:
    """Fit ENN parameters using Rust backend if available, otherwise Python."""
    from .enn_class import EpistemicNearestNeighbors as PyENN
    from .enn_params import ENNParams as PyENNParams

    if not isinstance(model, PyENN):
        raise TypeError(f"Expected EpistemicNearestNeighbors, got {type(model)}")

    # Use Python fallback if Rust is not available or model doesn't have Rust backend
    if _rust_enn_fit is None or model.rust_backend is None or RustENNParams is None:
        return _enn_fit_python(
            model,
            k=k,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
            rng=rng,
            params_warm_start=params_warm_start,
            infer_aleatoric_variance_scale=infer_aleatoric_variance_scale,
        )

    # Convert Python RNG to seed for Rust RNG
    seed = int(rng.integers(0, 2**63 - 1))

    # Convert warm-start params if provided
    rust_warm_start = None
    if params_warm_start is not None:
        rust_warm_start = RustENNParams(
            params_warm_start.k_num_neighbors,
            params_warm_start.epistemic_variance_scale,
            params_warm_start.aleatoric_variance_scale,
        )

    # Call Rust function
    rust_result = _rust_enn_fit(
        model.rust_backend,
        k,
        num_fit_candidates,
        num_fit_samples,
        seed,
        rust_warm_start,
        infer_aleatoric_variance_scale,
    )

    # Convert result back to Python ENNParams
    return PyENNParams(
        k_num_neighbors=rust_result.k_num_neighbors,
        epistemic_variance_scale=rust_result.epistemic_variance_scale,
        aleatoric_variance_scale=rust_result.aleatoric_variance_scale,
    )
