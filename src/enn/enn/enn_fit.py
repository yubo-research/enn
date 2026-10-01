from __future__ import annotations

from typing import Any

import numpy as np

from enn._rust import subsample_loglik as _rust_subsample_loglik

from .add_token import ENNAddToken


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
    ``model.add``. That token is consumed here, so a stale, foreign, or
    repeated token cannot be told. The first incremental call freezes ``k``
    and the seed drawn from ``rng`` on the model. A later incremental call
    must draw that same seed and pass that same ``k``. Any other pair raises
    ``ValueError`` instead of being ignored.
    """
    from .enn_class import EpistemicNearestNeighbors as PyENN
    from .enn_fitter import ENNStatefulFitter

    if not isinstance(model, PyENN):
        raise TypeError(f"Expected EpistemicNearestNeighbors, got {type(model)}")

    if incremental is None:
        fitter = ENNStatefulFitter(k=k, rng=rng)
        x_all, y_all, yvar_all = model.train_rows_at(list(range(len(model))))
        y_bounds = np.asarray(model.rust_backend.y_bounds, dtype=float)
        fitter.tell(x_all, y_all, yvar_all, y_bounds=y_bounds)
    else:
        if not isinstance(incremental, ENNAddToken):
            raise TypeError(
                "incremental must be the token returned by model.add, "
                f"got {type(incremental).__name__}"
            )
        proposed_seed = int(rng.integers(0, 2**63 - 1))
        fitter = getattr(model, "_incremental_fitter", None)
        if fitter is not None and (
            int(k) != fitter.k or proposed_seed != fitter.seed
        ):
            raise ValueError(
                "incremental enn_fit freezes k and the fitter seed on the first call; "
                f"frozen k={fitter.k}, frozen seed={fitter.seed}, "
                f"got k={int(k)}, seed={proposed_seed}"
            )
        x_delta, y_delta, yvar_delta = incremental.take(model)
        if fitter is None:
            fitter = ENNStatefulFitter(k=k, seed=proposed_seed)
            model._incremental_fitter = fitter
        y_bounds = np.asarray(model.rust_backend.y_bounds, dtype=float)
        fitter.tell(x_delta, y_delta, yvar_delta, y_bounds=y_bounds)

    return fitter.ask(
        model,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        params_warm_start=params_warm_start,
    )
