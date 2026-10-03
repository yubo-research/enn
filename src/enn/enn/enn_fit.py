from __future__ import annotations

from typing import Any

import numpy as np

from enn._rust import ENNAddToken
from enn._rust import enn_fit_incremental as _rust_enn_fit_incremental
from enn._rust import subsample_loglik as _rust_subsample_loglik

from .enn_fitter import ENNStatefulFitter, py_params, rust_params


def subsample_loglik(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    paramss: list[Any],
    P: int | None = None,
    rng: Any,
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
    model: Any,
    *,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int | None = None,
    rng: Any,
    params_warm_start: Any | None = None,
    incremental: ENNAddToken | None = None,
) -> Any:
    """Fit ENN hyperparameters via ENNStatefulFitter tell/ask.

    Batch mode (``incremental`` is None): tell the full model and ask once.
    ``k`` and ``rng`` build a new fitter on every call.

    Incremental mode: ``incremental`` must be the token just returned by
    ``model.add``. The Rust model consumes that token and tells only the rows
    of that ``add``, so a stale, foreign, or repeated token cannot be told.
    The first incremental call freezes ``k`` and the seed drawn from ``rng``
    on the Rust model. A later incremental call must draw that same seed and
    pass that same ``k``. Any other pair raises ``ValueError``.
    """
    from .enn_class import EpistemicNearestNeighbors as PyENN

    if not isinstance(model, PyENN):
        raise TypeError(f"Expected EpistemicNearestNeighbors, got {type(model)}")

    if incremental is None:
        fitter = ENNStatefulFitter(k=k, rng=rng)
        x_all, y_all, yvar_all = model.train_rows_at(list(range(len(model))))
        y_bounds = np.asarray(model.rust_backend.y_bounds, dtype=float)
        fitter.tell(x_all, y_all, yvar_all, y_bounds=y_bounds)
        return fitter.ask(
            model,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
            params_warm_start=params_warm_start,
        )
    if not isinstance(incremental, ENNAddToken):
        raise TypeError(
            "incremental must be the token returned by model.add, "
            f"got {type(incremental).__name__}"
        )
    rust_result = _rust_enn_fit_incremental(
        model.rust_backend,
        incremental,
        int(k),
        int(rng.integers(0, 2**63 - 1)),
        num_fit_candidates=None if num_fit_candidates is None else int(num_fit_candidates),
        num_fit_samples=num_fit_samples,
        params_warm_start=rust_params(params_warm_start),
    )
    return py_params(rust_result)
