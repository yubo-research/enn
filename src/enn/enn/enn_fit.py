from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from enn._rust import ENNParams as RustENNParams
from enn._rust import enn_fit as _rust_enn_fit
from enn._rust import subsample_loglik as _rust_subsample_loglik

if TYPE_CHECKING:
    from numpy.random import Generator

    from .enn_class import EpistemicNearestNeighbors
    from .enn_params import ENNParams


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
    """Compute subsample log-likelihood using Rust backend."""
    from .enn_class import EpistemicNearestNeighbors as PyENN

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)

    if not isinstance(model, PyENN):
        raise TypeError(f"Expected EpistemicNearestNeighbors, got {type(model)}")

    seed = int(rng.integers(0, 2**63 - 1))

    k_values = [p.k_num_neighbors for p in paramss]
    epi_scales = [p.epistemic_variance_scale for p in paramss]
    ale_scales = [p.aleatoric_variance_scale for p in paramss]

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
    """Fit ENN parameters using Rust backend."""
    from .enn_class import EpistemicNearestNeighbors as PyENN
    from .enn_params import ENNParams as PyENNParams

    if not isinstance(model, PyENN):
        raise TypeError(f"Expected EpistemicNearestNeighbors, got {type(model)}")

    seed = int(rng.integers(0, 2**63 - 1))

    rust_warm_start = None
    if params_warm_start is not None:
        rust_warm_start = RustENNParams(
            params_warm_start.k_num_neighbors,
            params_warm_start.epistemic_variance_scale,
            params_warm_start.aleatoric_variance_scale,
        )

    rust_result = _rust_enn_fit(
        model.rust_backend,
        k,
        num_fit_candidates,
        num_fit_samples,
        seed,
        rust_warm_start,
        infer_aleatoric_variance_scale,
    )

    return PyENNParams(
        k_num_neighbors=rust_result.k_num_neighbors,
        epistemic_variance_scale=rust_result.epistemic_variance_scale,
        aleatoric_variance_scale=rust_result.aleatoric_variance_scale,
    )
