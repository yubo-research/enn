from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from enn._rust import subsample_loglik as _rust_subsample_loglik

if TYPE_CHECKING:
    from numpy.random import Generator

    from .enn_class import EpistemicNearestNeighbors
    from .enn_fitter import ENNStatefulFitter
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


@dataclass(frozen=True)
class ENNIncrementalDelta:
    """Rows just appended via ``model.add`` plus the fitter tracking y_std."""

    fitter: ENNStatefulFitter
    x: np.ndarray
    y: np.ndarray
    yvar: np.ndarray | None = None


def enn_fit(
    model: EpistemicNearestNeighbors,
    *,
    k: int,
    num_fit_candidates: int,
    num_fit_samples: int = 10,
    rng: Generator,
    params_warm_start: ENNParams | None = None,
    incremental: ENNIncrementalDelta | None = None,
) -> ENNParams:
    """Fit ENN hyperparameters via ENNStatefulFitter tell/ask.

    Batch mode (``incremental`` is None): tell the full model and ask once.

    Incremental mode: tell only the delta rows that were just appended via
    ``model.add``, then ask — same rhythm as ``ENNStatefulFitter.ask``.
    """
    from .enn_class import EpistemicNearestNeighbors as PyENN
    from .enn_fitter import ENNStatefulFitter

    if not isinstance(model, PyENN):
        raise TypeError(f"Expected EpistemicNearestNeighbors, got {type(model)}")

    if incremental is None:
        fitter = ENNStatefulFitter(k=k, rng=rng)
        x_all, y_all, yvar_all = model.train_rows_at(list(range(len(model))))
        fitter.tell(x_all, y_all, yvar_all)
    else:
        fitter = incremental.fitter
        fitter.tell(incremental.x, incremental.y, incremental.yvar)

    return fitter.ask(
        model,
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        params_warm_start=params_warm_start,
    )
